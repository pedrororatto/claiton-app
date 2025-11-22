import os
import sys
import json
import requests
from typing import List, Dict, Tuple
from dotenv import load_dotenv
load_dotenv()
# CONFIGS
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX"))
TOP_P = float(os.getenv("OLLAMA_TOP_P"))
K_JURIS = int(os.getenv("K_JURIS"))
K_LEI = int(os.getenv("K_LEI"))

# Imports corrigidos (LangChain v0.2+)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Use o MESMO modelo de embeddings da indexação
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
# Forçar CPU para contornar incompatibilidade CUDA sm_61
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./vectordb/chroma")
JURIS_COLLECTION = "jurisprudencia_br_v1"
LEI_COLLECTION = "legislacao_codigo_penal"

SYSTEM_INSTRUCTIONS = """
Você é um assistente jurídico especializado em Direito Penal brasileiro.

### Objetivo:
Responder perguntas sobre crimes e infrações penais de forma **correta, objetiva e compreensível até para leigos**.

### Diretrizes:
- Baseie-se apenas nas leis e jurisprudências brasileiras.
- **Não invente** artigos, súmulas ou decisões.
- Se os "Contextos Recuperados" não forem suficientes, diga claramente que não é possível concluir.
- Quando houver **termos jurídicos difíceis**, explique-os **brevemente em linguagem simples**, antes da resposta jurídica.
- Depois dessa explicação, apresente a **resposta técnica resumida** (3 a 6 frases), de forma direta e impessoal.
- Cite as fontes ao final no formato: [Fonte {N} – {source_meta}].
- Evite opiniões pessoais e especulações.

### Estrutura sugerida da resposta:
1. (Opcional) Explicação simples de termos jurídicos difíceis.  
2. Resposta jurídica objetiva e fundamentada.  
3. Fontes citadas no formato indicado.
"""

PROMPT_TEMPLATE = """
[SISTEMA]
{system_instructions}

[PERGUNTA DO USUÁRIO]
{question}

[CONTEXTOS RECUPERADOS]
{contexts}

[INSTRUÇÕES DE SAÍDA]
- Responda em português do Brasil.
- Seja conciso, técnico e juridicamente preciso.
- Cite as fontes no final no formato [Fonte N – {{source_meta}}].
"""

def load_vectorstores():
    juris = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=EMBEDDINGS,
        collection_name=JURIS_COLLECTION
    )
    lei = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=EMBEDDINGS,
        collection_name=LEI_COLLECTION
    )
    return juris, lei

def dual_retrieve(question: str, k_juris=3, k_lei=3) -> List[Dict]:
    juris, lei = load_vectorstores()
    docs_juris = juris.similarity_search_with_score(question, k=k_juris)
    docs_lei = lei.similarity_search_with_score(question, k=k_lei)

    # Normalize results into a list of dicts: content, meta, score, origem
    results = []
    for doc, score in docs_juris:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
            "origem": "jurisprudencia"
        })
    for doc, score in docs_lei:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
            "origem": "legislacao"
        })
    # Opcional: reordenar por score ascendente
    results.sort(key=lambda x: x["score"])
    return results

def format_contexts(chunks: List[Dict], max_chars: int = 6000) -> Tuple[str, List[Dict]]:
    formatted = []
    used = []
    total = 0
    for i, ch in enumerate(chunks, start=1):
        doc_id = ch["metadata"].get("id") or ch["metadata"].get("source") or ch["metadata"].get("file") or "doc"
        titulo = ch["metadata"].get("titulo") or ch["metadata"].get("title") or doc_id
        header = f"[Fonte {i}] id={doc_id}, origem={ch['origem']}, score={ch['score']:.4f}, titulo={titulo}\n"
        block = header + ch["content"].strip() + "\n"
        if total + len(block) > max_chars:
            break
        formatted.append(block)
        used.append(ch)
        total += len(block)
    return "\n".join(formatted), used

def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_ctx": NUM_CTX
        },
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def answer(question: str):
    retrieved = dual_retrieve(question, k_juris=K_JURIS, k_lei=K_LEI)
    contexts_str, used = format_contexts(retrieved)
    prompt = PROMPT_TEMPLATE.format(
        system_instructions=SYSTEM_INSTRUCTIONS,
        question=question.strip(),
        contexts=contexts_str if contexts_str else "(nenhum contexto recuperado)"
    )

    response = call_ollama(prompt)
    print("\nRESPOSTA:")
    print(response)
    print("\nFONTES:")
    if used:
        for i, ch in enumerate(used, start=1):
            meta = ch["metadata"]
            src = meta.get("id") or meta.get("source") or meta.get("file") or "doc"
            titulo = meta.get("titulo") or meta.get("title") or src
            print(f"[Fonte {i}] {titulo} ({src}) - origem={ch['origem']} score={ch['score']:.4f}")
    else:
        print("Nenhuma fonte utilizada (sem contexto).")


def answer_question(question: str, max_response_length: int = None) -> Tuple[str, List[Dict]]:
    """
    Responde uma pergunta usando RAG.
    
    Args:
        question: Pergunta do usuário
        max_response_length: Limite opcional de caracteres para a resposta (útil para WhatsApp)
    
    Returns:
        Tupla (resposta, lista_de_fontes)
    """
    try:
        # Retrieve
        retrieved = dual_retrieve(question, k_juris=K_JURIS, k_lei=K_LEI)

        if not retrieved:
            return "Não encontrei informações relevantes sobre isso. Pode reformular a pergunta?", []

        contexts_str, used = format_contexts(retrieved)
        prompt = PROMPT_TEMPLATE.format(
            system_instructions=SYSTEM_INSTRUCTIONS,
            question=question.strip(),
            contexts=contexts_str if contexts_str else "(nenhum contexto recuperado)"
        )
        response = call_ollama(prompt)

        # Truncar resposta se max_response_length foi especificado
        if max_response_length and len(response) > max_response_length:
            # Tentar truncar em uma frase completa
            resposta_truncada = response[:max_response_length - 50]
            ultimo_ponto = resposta_truncada.rfind('.')
            ultima_quebralinha = resposta_truncada.rfind('\n')
            ponto_corte = max(ultimo_ponto, ultima_quebralinha)
            
            if ponto_corte > max_response_length * 0.7:  # Se encontrou um ponto razoavelmente próximo
                response = resposta_truncada[:ponto_corte + 1]
            else:
                response = resposta_truncada
            
            response += "\n\n⚠️ *Mensagem truncada devido ao limite de caracteres.*"

        fontes = []
        for ch in used:
            meta = ch["metadata"]
            fontes.append({
                "titulo": meta.get("titulo") or meta.get("title") or meta.get("id") or "Documento",
                "id": meta.get("id") or meta.get("source") or meta.get("file") or "N/A",
                "origem": ch["origem"],
                "score": ch["score"],
                "text": ch["content"]
            })

        return response, fontes

    except Exception as e:
        print(f"[ERRO em answer_question] {e}")
        import traceback
        traceback.print_exc()
        return "Erro ao processar sua pergunta. Tente novamente.", []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python rag_cli.py \"sua pergunta em PT-BR\"")
        sys.exit(1)
    question = sys.argv[1]
    answer(question)
