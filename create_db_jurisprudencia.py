import json
import os
from pathlib import Path
from typing import List, Dict, Any, Iterable
from dotenv import load_dotenv

# Use o pacote novo do Chroma para LangChain
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
EMBEDDING_MODEL_NAME = (os.getenv("EMBED_MODEL_NAME"))
print(EMBEDDING_MODEL_NAME)
# Instância global, forçando CPU
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

DIR_CHUNKS = Path("dados_sanitizados/chunks")
CHROMA_DB_DIR = Path("./vectordb/chroma")
CHROMA_COLLECTION = "jurisprudencia_br_v1"

def carregar_chunks(dir_chunks: Path) -> Iterable[Dict[str, Any]]:
    for json_file in dir_chunks.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
        except Exception as e:
            print(f"[ERRO] {json_file.name}: {e}")

def chunk_to_document(item):
    page_content = "passage: " + (item.get("texto", "") or "")
    metadata = {k: v for k, v in {
        "chunk_id": item.get("chunk_id"),
        "crime": item.get("crime"),
        "tribunal": item.get("tribunal"),
        "orgao_julgador": item.get("orgao_julgador"),
        "data": item.get("data"),
        "ementa": item.get("ementa"),
        "fonte": item.get("fonte"),
        "arquivo_origem": item.get("arquivo_origem"),
    }.items() if v is not None}
    return Document(page_content=page_content, metadata=metadata)

def indexar_chunks_em_chroma():
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    # REUTILIZA a instância global embeddings (CPU). Não recrie sem device="cpu"
    vectordb = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )

    documentos: List[Document] = []
    total_items = 0

    for item in carregar_chunks(DIR_CHUNKS):
        total_items += 1
        doc = chunk_to_document(item)
        if doc.page_content and doc.page_content.strip():
            documentos.append(doc)

        if len(documentos) >= 512:
            vectordb.add_documents(documentos)
            print(f"✓ Inseridos {len(documentos)} documentos (parcial).")
            documentos = []

    if documentos:
        vectordb.add_documents(documentos)

    print("=" * 60)
    print(f"✅ Indexação concluída!")
    print(f"📊 Total de items: {total_items}")
    print(f"📁 Base vetorial: {CHROMA_DB_DIR}")
    print(f"🗂️  Coleção: {CHROMA_COLLECTION}")
    print("=" * 60)

def teste_busca(query: str, k: int = 3):
    # REUTILIZA a instância global embeddings (CPU)
    vectordb = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )
    # Para e5, prefira prefixar a query:
    q = "query: " + query
    docs = vectordb.similarity_search(q, k=k)
    print(f"\n🔍 Consulta: {query}\nTop {k} resultados:\n")
    for i, d in enumerate(docs, 1):
        m = d.metadata
        print(f"[{i}] {m.get('chunk_id')} | {m.get('tribunal')} | {m.get('crime')}")
        print(f"    {m.get('orgao_julgador')} | {m.get('data')}")
        if m.get("ementa"):
            print(f"    Ementa: {m.get('ementa')[:120]}...")
        print(f"    Fonte: {m.get('fonte')}")
        print(f"    Trecho: {d.page_content[:200].replace(chr(10), ' ')}...\n")

if __name__ == "__main__":
    # Descomente para reindexar (só precisa rodar 1x)
    indexar_chunks_em_chroma()

    # Testes de busca
    teste_busca("princípio da insignificância furto", k=3)
    teste_busca("tráfico regime semiaberto art 35", k=3)