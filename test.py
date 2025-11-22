"""
test_llms.py - Teste de LLMs para CLAITON TCC

Testa diferentes modelos de LLM usando o banco vetorial já criado,
calculando métricas de desempenho (Precision, Recall, F1-Score).

Pré-requisito: Banco vetorial já criado com o embedding desejado.

Uso:
    python test_llms.py

Saída:
    - CSV detalhado: resultados_llm/resultados_YYYYMMDD_HHMMSS.csv
    - CSV resumido: resultados_llm/resultados_YYYYMMDD_HHMMSS_resumo.csv
"""

import os
import sys
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
from dotenv import load_dotenv

# Importar funções do rag_core
from rag_core import (
    dual_retrieve,
    format_contexts,
    call_ollama,
    SYSTEM_INSTRUCTIONS,
    PROMPT_TEMPLATE,
    K_JURIS,
    K_LEI,
    EMBED_MODEL_NAME
)

load_dotenv()

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Modelos de LLM a testar (ajuste conforme seus modelos Ollama)
LLM_MODELS = {
    "llama3_3b_q8": "llama3:3b-instruct-q8_0",
    "llama3_8b_q6": "llama3:8b-instruct-q6_K",
}

# Arquivo de perguntas e gabarito
PERGUNTAS_CSV = "perguntas_gabarito.csv"

# Diretório de saída
OUTPUT_DIR = Path("resultados_llm")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# FUNÇÕES DE AVALIAÇÃO
# ============================================================================

def extract_retrieved_ids(retrieved_docs: List[Dict]) -> tuple[Set[str], Set[str]]:
    """
    Extrai IDs dos documentos recuperados, separados por origem.
    
    Returns:
        (set_artigos, set_juris)
    """
    artigos = set()
    juris = set()
    
    for doc in retrieved_docs:
        origem = doc["origem"]
        metadata = doc["metadata"]
        
        if origem == "legislacao":
            # Para legislação, usar o número do artigo
            artigo_num = metadata.get("artigo", "")
            if artigo_num:
                artigos.add(str(artigo_num))
        elif origem == "jurisprudencia":
            # Para jurisprudência, usar o chunk_id
            chunk_id = metadata.get("id", "")
            if chunk_id:
                juris.add(chunk_id)
    
    return artigos, juris


def calculate_metrics(retrieved: Set[str], relevant: Set[str]) -> Dict[str, float]:
    """
    Calcula Precision, Recall e F1-Score.
    
    Args:
        retrieved: IDs dos documentos recuperados
        relevant: IDs dos documentos relevantes (gabarito)
    
    Returns:
        Dict com precision, recall, f1
    """
    if not retrieved and not relevant:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if not relevant:
        # Se não há gabarito, não podemos avaliar
        return {"precision": None, "recall": None, "f1": None}
    
    tp = len(retrieved & relevant)  # True Positives
    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================================
# CARREGAMENTO DE PERGUNTAS
# ============================================================================

def load_questions(csv_path: str) -> List[Dict]:
    """
    Carrega perguntas e gabarito do CSV.
    
    Formato esperado:
    id_pergunta,pergunta,resposta_esperada,artigos_relevantes,juris_relevantes
    
    artigos_relevantes: separados por ; (ex: "121;155")
    juris_relevantes: separados por ; (ex: "chunk_001;chunk_045")
    """
    questions = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            artigos_str = row.get("artigos_relevantes", "").strip()
            juris_str = row.get("juris_relevantes", "").strip()
            
            artigos_relevantes = set(artigos_str.split(";")) if artigos_str else set()
            juris_relevantes = set(juris_str.split(";")) if juris_str else set()
            
            # Limpar espaços em branco
            artigos_relevantes = {a.strip() for a in artigos_relevantes if a.strip()}
            juris_relevantes = {j.strip() for j in juris_relevantes if j.strip()}
            
            questions.append({
                "id": row["id_pergunta"],
                "pergunta": row["pergunta"],
                "resposta_esperada": row.get("resposta_esperada", ""),
                "artigos_relevantes": artigos_relevantes,
                "juris_relevantes": juris_relevantes
            })
    
    return questions


# ============================================================================
# TESTE DE UMA PERGUNTA
# ============================================================================

def test_single_question(
    question_data: Dict,
    llm_name: str,
    llm_model: str,
    retrieved_docs: List[Dict]
) -> Dict:
    """
    Testa uma única pergunta com um LLM específico.
    
    Args:
        question_data: Dados da pergunta
        llm_name: Nome do LLM
        llm_model: Modelo do LLM
        retrieved_docs: Documentos já recuperados (para reutilizar)
    
    Returns:
        Dict com resultados e métricas
    """
    pergunta = question_data["pergunta"]
    
    # Extrair IDs recuperados
    artigos_retrieved, juris_retrieved = extract_retrieved_ids(retrieved_docs)
    
    # Calcular métricas de retrieval
    metrics_lei = calculate_metrics(artigos_retrieved, question_data["artigos_relevantes"])
    metrics_juris = calculate_metrics(juris_retrieved, question_data["juris_relevantes"])
    
    # Formatar contextos usando função do rag_core
    contexts_str, used_docs = format_contexts(retrieved_docs)
    
    # Montar prompt usando template do rag_core
    prompt = PROMPT_TEMPLATE.format(
        system_instructions=SYSTEM_INSTRUCTIONS,
        question=pergunta,
        contexts=contexts_str if contexts_str else "(nenhum contexto recuperado)"
    )
    
    # Medir tempo de geração
    start_generation = time.time()
    try:
        # Usar função call_ollama do rag_core, mas com modelo específico
        resposta = call_ollama(prompt, model=llm_model)
    except Exception as e:
        resposta = f"[ERRO] {str(e)}"
    generation_time = time.time() - start_generation
    
    return {
        "id_pergunta": question_data["id"],
        "pergunta": pergunta,
        "embedding_model": EMBED_MODEL_NAME,
        "llm_name": llm_name,
        "llm_model": llm_model,
        "resposta_gerada": resposta,
        "resposta_esperada": question_data["resposta_esperada"],
        "artigos_retrieved": ";".join(sorted(artigos_retrieved)),
        "artigos_relevantes": ";".join(sorted(question_data["artigos_relevantes"])),
        "juris_retrieved": ";".join(sorted(juris_retrieved)),
        "juris_relevantes": ";".join(sorted(question_data["juris_relevantes"])),
        "precision_lei": metrics_lei["precision"],
        "recall_lei": metrics_lei["recall"],
        "f1_lei": metrics_lei["f1"],
        "precision_juris": metrics_juris["precision"],
        "recall_juris": metrics_juris["recall"],
        "f1_juris": metrics_juris["f1"],
        "generation_time": generation_time,
        "num_docs_retrieved": len(retrieved_docs),
        "num_docs_used": len(used_docs)
    }


# ============================================================================
# EXECUÇÃO DOS TESTES
# ============================================================================

def run_all_tests():
    """Executa todos os testes e salva resultados."""
    print("=" * 80)
    print("TESTE DE LLMs - CLAITON TCC")
    print("=" * 80)
    print(f"Embedding usado: {EMBED_MODEL_NAME}")
    print(f"LLMs a testar: {', '.join(LLM_MODELS.keys())}")
    print(f"K_JURIS={K_JURIS}, K_LEI={K_LEI}")
    print("=" * 80)
    
    # Carregar perguntas
    print(f"\n📖 Carregando perguntas de {PERGUNTAS_CSV}...")
    if not Path(PERGUNTAS_CSV).exists():
        print(f"❌ ERRO: Arquivo {PERGUNTAS_CSV} não encontrado!")
        print("Crie o arquivo com o formato:")
        print("id_pergunta,pergunta,resposta_esperada,artigos_relevantes,juris_relevantes")
        sys.exit(1)
    
    questions = load_questions(PERGUNTAS_CSV)
    print(f"✅ {len(questions)} perguntas carregadas.")
    
    # Preparar arquivo de saída
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"resultados_{timestamp}.csv"
    
    # Cabeçalho do CSV
    fieldnames = [
        "id_pergunta", "pergunta", "embedding_model",
        "llm_name", "llm_model", "resposta_gerada", "resposta_esperada",
        "artigos_retrieved", "artigos_relevantes", "juris_retrieved", "juris_relevantes",
        "precision_lei", "recall_lei", "f1_lei",
        "precision_juris", "recall_juris", "f1_juris",
        "generation_time", "num_docs_retrieved", "num_docs_used"
    ]
    
    # Cache de retrieval (fazer uma vez por pergunta, reutilizar para todos os LLMs)
    print(f"\n🔍 Fazendo retrieval para todas as perguntas...")
    retrieved_cache = {}
    for i, question in enumerate(questions, 1):
        print(f"   [{i}/{len(questions)}] {question['id']}: {question['pergunta'][:60]}...")
        try:
            retrieved_docs = dual_retrieve(question['pergunta'], k_juris=K_JURIS, k_lei=K_LEI)
            retrieved_cache[question['id']] = retrieved_docs
        except Exception as e:
            print(f"      ❌ ERRO no retrieval: {e}")
            retrieved_cache[question['id']] = []
    
    print(f"✅ Retrieval concluído para {len(retrieved_cache)} perguntas.\n")
    
    # Executar testes
    total_tests = len(LLM_MODELS) * len(questions)
    current_test = 0
    
    print(f"🚀 Iniciando {total_tests} testes de geração...")
    print(f"   {len(LLM_MODELS)} LLMs × {len(questions)} perguntas\n")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for llm_name, llm_model in LLM_MODELS.items():
            print(f"\n{'=' * 80}")
            print(f"🤖 Testando LLM: {llm_name} ({llm_model})")
            print(f"{'=' * 80}")
            
            for i, question in enumerate(questions, 1):
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] Pergunta {i}/{len(questions)}: {question['id']}")
                print(f"   {question['pergunta'][:80]}...")
                
                try:
                    retrieved_docs = retrieved_cache.get(question['id'], [])
                    
                    if not retrieved_docs:
                        print(f"   ⚠️  Nenhum documento recuperado (pulando)")
                        continue
                    
                    result = test_single_question(
                        question, llm_name, llm_model, retrieved_docs
                    )
                    writer.writerow(result)
                    f.flush()  # Salvar imediatamente
                    
                    # Exibir métricas
                    p_lei = result['precision_lei'] if result['precision_lei'] is not None else 0.0
                    r_lei = result['recall_lei'] if result['recall_lei'] is not None else 0.0
                    f1_lei = result['f1_lei'] if result['f1_lei'] is not None else 0.0
                    p_juris = result['precision_juris'] if result['precision_juris'] is not None else 0.0
                    r_juris = result['recall_juris'] if result['recall_juris'] is not None else 0.0
                    f1_juris = result['f1_juris'] if result['f1_juris'] is not None else 0.0
                    
                    print(f"   ✅ P_lei={p_lei:.2f} R_lei={r_lei:.2f} F1_lei={f1_lei:.2f} | "
                          f"P_juris={p_juris:.2f} R_juris={r_juris:.2f} F1_juris={f1_juris:.2f} | "
                          f"Tempo={result['generation_time']:.2f}s")
                
                except Exception as e:
                    print(f"   ❌ ERRO: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print(f"✅ TESTES CONCLUÍDOS!")
    print(f"📁 Resultados salvos em: {output_file}")
    print(f"{'=' * 80}\n")
    
    return output_file


# ============================================================================
# ANÁLISE DE RESULTADOS
# ============================================================================

def analyze_results(results_file: Path):
    """Gera resumo agregado dos resultados."""
    print(f"\n📊 Analisando resultados de {results_file}...")
    
    # Carregar resultados
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    if not results:
        print("❌ Nenhum resultado encontrado no arquivo.")
        return None
    
    # Agrupar por LLM
    from collections import defaultdict
    groups = defaultdict(list)
    
    for row in results:
        key = row["llm_name"]
        groups[key].append(row)
    
    # Calcular médias
    summary = []
    for llm_name, rows in groups.items():
        n = len(rows)
        
        # Filtrar valores None
        def safe_mean(values):
            valid = [float(v) for v in values if v and v != 'None' and v != '']
            return sum(valid) / len(valid) if valid else 0.0
        
        summary.append({
            "embedding": rows[0]["embedding_model"],
            "llm": llm_name,
            "llm_model": rows[0]["llm_model"],
            "n_perguntas": n,
            "precision_lei_mean": safe_mean([r["precision_lei"] for r in rows]),
            "recall_lei_mean": safe_mean([r["recall_lei"] for r in rows]),
            "f1_lei_mean": safe_mean([r["f1_lei"] for r in rows]),
            "precision_juris_mean": safe_mean([r["precision_juris"] for r in rows]),
            "recall_juris_mean": safe_mean([r["recall_juris"] for r in rows]),
            "f1_juris_mean": safe_mean([r["f1_juris"] for r in rows]),
            "generation_time_mean": safe_mean([r["generation_time"] for r in rows]),
        })
    
    # Salvar resumo
    summary_file = results_file.with_name(results_file.stem + "_resumo.csv")
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "embedding", "llm", "llm_model", "n_perguntas",
            "precision_lei_mean", "recall_lei_mean", "f1_lei_mean",
            "precision_juris_mean", "recall_juris_mean", "f1_juris_mean",
            "generation_time_mean"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    
    # Imprimir tabela
    print("\n" + "=" * 120)
    print("RESUMO DOS RESULTADOS")
    print("=" * 120)
    print(f"{'Embedding':<40} {'LLM':<20} {'N':<5} {'P_Lei':<8} {'R_Lei':<8} {'F1_Lei':<8} "
          f"{'P_Juris':<8} {'R_Juris':<8} {'F1_Juris':<8} {'Tempo(s)':<10}")
    print("-" * 120)
    
    for row in summary:
        emb_short = row['embedding'].split('/')[-1][:38]  # Encurtar nome do embedding
        print(f"{emb_short:<40} {row['llm']:<20} {row['n_perguntas']:<5} "
              f"{row['precision_lei_mean']:<8.4f} {row['recall_lei_mean']:<8.4f} {row['f1_lei_mean']:<8.4f} "
              f"{row['precision_juris_mean']:<8.4f} {row['recall_juris_mean']:<8.4f} {row['f1_juris_mean']:<8.4f} "
              f"{row['generation_time_mean']:<10.2f}")
    
    print("=" * 120)
    print(f"\n✅ Resumo salvo em: {summary_file}\n")
    
    return summary_file


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🎓 CLAITON TCC - Teste de LLMs\n")
    
    # Verificar se o embedding está configurado
    if not EMBED_MODEL_NAME:
        print("❌ ERRO: EMBED_MODEL_NAME não configurado no .env")
        sys.exit(1)
    
    # Executar testes
    results_file = run_all_tests()
    
    # Analisar resultados
    analyze_results(results_file)
    
    print("🎉 Processo concluído com sucesso!\n")