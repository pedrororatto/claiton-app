import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configurações
CODIGO_PENAL_JSON = Path("dados_sanitizados/codigo_penal/codigo_penal_estruturado.json")
CHROMA_DB_DIR = Path("./vectordb/chroma")

def carregar_codigo_penal(caminho_json):
    with open(caminho_json, 'r', encoding='utf-8') as f:
        return json.load(f)

def criar_documento_artigo(tema, artigo, metadata_geral):
    texto_completo = f"TEMA: {tema}\n\n"
    texto_completo += f"Artigo {artigo['artigo']}: {artigo['titulo']}\n\n"
    texto_completo += f"Descrição: {artigo['texto']}\n\n"
    texto_completo += f"Pena: {artigo['pena']}\n\n"

    paragrafos = []
    for key, value in artigo.items():
        if key.startswith('paragrafo_') and value:
            num_paragrafo = key.replace('paragrafo_', '').replace('_', '-')
            paragrafos.append(f"§{num_paragrafo}: {value}")
    if paragrafos:
        texto_completo += "Parágrafos:\n" + "\n".join(paragrafos) + "\n\n"

    if 'nota' in artigo and artigo['nota']:
        texto_completo += f"Nota: {artigo['nota']}\n"

    metadata = {
        "tipo": "legislacao",
        "fonte": metadata_geral['fonte'],
        "decreto_lei": metadata_geral['decreto_lei'],
        "url_oficial": metadata_geral['url_oficial'],
        "tema": tema,
        "artigo": artigo['artigo'],
        "titulo": artigo['titulo'],
        "pena": artigo['pena'],
    }
    return texto_completo, metadata

def indexar_codigo_penal():
    print("🔄 Iniciando indexação do Código Penal...")
    print(f"📖 Carregando {CODIGO_PENAL_JSON}...")
    codigo_penal = carregar_codigo_penal(CODIGO_PENAL_JSON)

    print(f"🔗 Conectando ao ChromaDB em {CHROMA_DB_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    collection = client.get_or_create_collection(name="legislacao_codigo_penal")

    # PADRONIZAÇÃO: mesmo modelo da jurisprudência, em CPU
    print("🤖 Carregando modelo de embeddings (CPU)...")
    model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")

    documentos = []
    metadados = []
    ids = []

    contador = 0
    for tema_obj in codigo_penal['temas']:
        tema = tema_obj['tema']
        print(f"  📋 Processando tema: {tema}")
        for artigo in tema_obj['artigos']:
            texto, metadata = criar_documento_artigo(tema, artigo, codigo_penal['metadata'])
            documentos.append(texto)
            metadados.append(metadata)
            ids.append(f"legislacao_{tema.lower().replace(' ', '_')}_{artigo['artigo']}")
            contador += 1

    print(f"🧮 Gerando embeddings para {contador} artigos...")
    # E5 recomenda prefixo "passage: " nos documentos
    docs_e5 = [f"passage: {d}" for d in documentos]
    embeddings = model.encode(
        docs_e5,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    print("💾 Adicionando artigos ao ChromaDB...")
    collection.add(
        documents=documentos,
        embeddings=embeddings.tolist(),
        metadatas=metadados,
        ids=ids
    )

    print(f"\n✅ Indexação concluída!")
    print(f"📊 Total de artigos indexados: {contador}")
    print(f"📁 Banco de dados: {CHROMA_DB_DIR}")

    total_docs = collection.count()
    print(f"📈 Total de documentos no banco: {total_docs}")
    return contador

def testar_busca():
    print("\n🔍 Testando busca (direto na coleção)...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(name="legislacao_codigo_penal")

    model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")

    query1 = "Qual a pena para homicídio qualificado?"
    print(f"\n📝 Query: {query1}")
    # E5 recomenda prefixo "query: "
    q_emb = model.encode([f"query: {query1}"], convert_to_numpy=True)[0]
    results = collection.query(query_embeddings=[q_emb.tolist()], n_results=3)

    print("Resultados:")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        tipo = metadata.get('tipo', 'legislacao')
        print(f"\n  {i + 1}. [{tipo.upper()}]")
        print(f"     Artigo {metadata.get('artigo')}: {metadata.get('titulo')}")
        print(f"     Tema: {metadata.get('tema', 'N/A')}")
        print(f"     Trecho: {doc[:200]}...")

if __name__ == "__main__":
    total_indexado = indexar_codigo_penal()
    testar_busca()
    print("\n🎉 Processo concluído!")