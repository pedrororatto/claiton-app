# ⚖️ CLAITON - Assistente Jurídico Inteligente

## 📋 Índice

- [Descrição do Projeto](#descrição-do-projeto)
- [Arquitetura Técnica](#arquitetura-técnica)
- [Instalação e Configuração](#instalação-e-configuração)
- [Workflow Completo do Sistema](#workflow-completo-do-sistema)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Tecnologias, Algoritmos e Conceitos](#tecnologias-algoritmos-e-conceitos)
- [Contribuição](#contribuição)
- [Licença](#licença)

---

## 📖 Descrição do Projeto

### Visão Geral

**CLAITON** (Classificação Legal com Assistente Inteligente para Tratamento de Oportunidades Normativas) é um assistente jurídico inteligente especializado em **Direito Penal brasileiro**. O sistema utiliza técnicas avançadas de **RAG (Retrieval Augmented Generation)** para fornecer respostas precisas e fundamentadas sobre jurisprudência e legislação penal brasileira.

### Objetivo

O projeto visa democratizar o acesso à informação jurídica, permitindo que profissionais do direito, estudantes e cidadãos realizem consultas rápidas e precisas sobre:

- **Jurisprudência brasileira**: Precedentes e decisões dos tribunais superiores (STJ, STF) e regionais
- **Legislação penal**: Código Penal brasileiro e seus artigos estruturados

### Funcionalidades Principais

1. **Consulta Inteligente**: Realiza buscas semânticas em grandes volumes de documentos jurídicos
2. **Retrieval Híbrido**: Combina informações de jurisprudência e legislação para respostas completas
3. **Citações e Fontes**: Todas as respostas incluem referências às fontes consultadas com metadados completos
4. **Múltiplas Interfaces**:
   - Interface web interativa (Streamlit)
   - Bot WhatsApp para consultas via mensagens
5. **Processamento de Documentos**: Pipeline completo para extração, sanitização e indexação de PDFs jurídicos

### Contexto Acadêmico

Este projeto foi desenvolvido como **Trabalho de Conclusão de Curso (TCC) - 2025**, aplicando técnicas modernas de **Inteligência Artificial** e **Large Language Models (LLMs)** para resolver desafios reais na área jurídica. O sistema demonstra a aplicação prática de:

- Processamento de Linguagem Natural (NLP)
- Embeddings vetoriais para busca semântica
- Arquiteturas RAG para sistemas de perguntas e respostas
- Integração de múltiplas fontes de dados estruturadas

### Público-Alvo

- **Advogados e profissionais do direito** que necessitam de acesso rápido a jurisprudência
- **Estudantes de direito** em busca de precedentes e interpretações legais
- **Pesquisadores jurídicos** que precisam analisar grandes volumes de decisões
- **Cidadãos** interessados em compreender aspectos do direito penal brasileiro

---

## 🏗️ Arquitetura Técnica

### Sistema RAG (Retrieval Augmented Generation)

O CLAITON implementa uma arquitetura RAG híbrida que combina:

1. **Camada de Retrieval (Recuperação)**:
   - **ChromaDB**: Banco de dados vetorial para armazenamento de embeddings
   - **Duas coleções especializadas**:
     - `jurisprudencia_br_v1`: Decisões judiciais e acórdãos
     - `legislacao_codigo_penal`: Artigos do Código Penal brasileiro
   - **Modelo de Embeddings**: HuggingFace Transformers (E5-based) para representação semântica

2. **Camada de Generation (Geração)**:
   - **Ollama**: LLM local para geração de respostas contextualizadas
   - **Prompt Engineering**: Templates especializados para respostas jurídicas precisas
   - **Validação de Fontes**: Sistema garante que respostas sejam baseadas apenas em documentos recuperados

### Pipeline de Dados

```
PDFs Jurídicos → Extração → Sanitização → Chunking → Embeddings → ChromaDB
                                                                    ↓
Código Penal JSON → Estruturação → Embeddings → ChromaDB → Query do Usuário
                                                                    ↓
                                                          Retrieval Híbrido
                                                                    ↓
                                                          Geração de Resposta
```

### Componentes Principais

- **`rag_core.py`**: Motor RAG principal com retrieval dual e geração de respostas
- **`streamlit_app.py`**: Interface web com chat interativo
- **`whatssap_bot.py`**: Integração WhatsApp via Twilio
- **`sanitaze.py`**: Processamento e sanitização de PDFs jurídicos
- **`create_db_jurisprudencia.py`**: Indexação de jurisprudência
- **`create_db_cp.py`**: Indexação do Código Penal

---

## 🚀 Instalação e Configuração

### Pré-requisitos

Antes de começar, certifique-se de ter instalado:

- **Python 3.8 ou superior** (recomendado Python 3.10+)
- **pip** (gerenciador de pacotes Python)
- **Git** (para clonar o repositório)
- **Ollama** instalado e configurado localmente
- **ngrok** (para desenvolvimento do bot WhatsApp - opcional)
- Acesso à internet para download de modelos e dependências

### Passo 1: Clone do Repositório

```bash
git clone <repository-url>
cd claiton-app
```

### Passo 2: Criação do Ambiente Virtual

É **altamente recomendado** usar um ambiente virtual para isolar as dependências do projeto:

#### Linux/Mac:
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar o ambiente virtual
source venv/bin/activate
```

#### Windows:
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar o ambiente virtual
venv\Scripts\activate
```

Após ativar, você verá `(venv)` no início do seu prompt de terminal.

### Passo 3: Instalação de Dependências

Com o ambiente virtual ativado, instale todas as dependências:

```bash
# Atualizar pip (recomendado)
pip install --upgrade pip

# Instalar dependências do projeto
pip install -r requirements.txt
```

**Nota**: A instalação pode levar alguns minutos, especialmente ao baixar modelos de embeddings (sentence-transformers) e dependências do PyTorch.

### Passo 4: Verificação das Instalações

Verifique se as principais bibliotecas foram instaladas corretamente:

```bash
python -c "import langchain; import chromadb; import streamlit; print('✅ Dependências principais instaladas!')"
```

### Passo 5: Configuração do Ollama

1. **Instale o Ollama**:
   - Acesse: https://ollama.ai
   - Siga as instruções de instalação para seu sistema operacional

2. **Baixe um modelo LLM**:
```bash
# Opções de modelos (escolha um):
ollama pull llama2          # Modelo mais leve
ollama pull mistral         # Boa relação qualidade/performance
ollama pull llama2:13b      # Maior qualidade, mais recursos
ollama pull codellama       # Especializado em código/texto técnico
```

3. **Teste o Ollama**:
```bash
ollama run llama2 "Olá, você está funcionando?"
```

### Passo 6: Configuração das Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
touch .env  # Linux/Mac
# ou
type nul > .env  # Windows
```

Edite o arquivo `.env` com as seguintes configurações:

```env
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_TEMPERATURE=0.7
OLLAMA_NUM_CTX=4096
OLLAMA_TOP_P=0.9

# Vector Database
CHROMA_PATH=./vectordb/chroma
EMBED_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Retrieval Configuration
K_JURIS=3
K_LEI=3

# Twilio (apenas para WhatsApp Bot)
TWILIO_ACCOUNT_SID=seu-account-sid-aqui
TWILIO_AUTH_TOKEN=seu-auth-token-aqui
TWILIO_WHATSAPP_NUMBER=whatsapp:+5511999999999
```

**Importante**: Substitua os valores de `TWILIO_*` pelas suas credenciais reais se for usar o bot WhatsApp.

---

## 🔄 Workflow Completo do Sistema

Este guia detalha o fluxo completo de setup e uso do CLAITON, desde o processamento de documentos até a execução das interfaces.

### Visão Geral do Fluxo

```
1. Sanitização de PDFs → 2. Indexação Jurisprudência → 3. Indexação Código Penal → 4. Executar Interfaces
```

### Passo 1: Sanitização de Documentos PDF

Antes de indexar os documentos, é necessário processar e sanitizar os PDFs de jurisprudência:

1. **Prepare os PDFs**:
   - Coloque todos os PDFs de jurisprudência em um diretório
   - Exemplo: `/caminho/para/seu/diretorio/pdfs`

2. **Configure o diretório no código**:
   - Edite `sanitaze.py` e modifique a variável `DIRETORIO_PDFS` na função `main()`:
   ```python
   DIRETORIO_PDFS = '/caminho/para/seu/diretorio/pdfs'
   ```

3. **Execute a sanitização**:
   ```bash
   python sanitaze.py
   ```

   **O que este script faz**:
   - Extrai texto de todos os PDFs
   - Remove ruídos (copyright, headers, footers)
   - Extrai metadados (tribunal, processo, data, crime, etc.)
   - Gera chunks otimizados para busca vetorial
   - Salva em `dados_sanitizados/acordaos/` (JSONs estruturados)
   - Salva chunks em `dados_sanitizados/chunks/` (JSONs para indexação)

4. **Verifique os resultados**:
   - Confira os arquivos gerados em `dados_sanitizados/`
   - O script exibirá estatísticas ao final

### Passo 2: Indexação da Jurisprudência

Após sanitizar os documentos, indexe-os no ChromaDB:

```bash
python create_db_jurisprudencia.py
```

**O que este script faz**:
- Carrega todos os chunks de `dados_sanitizados/chunks/`
- Gera embeddings usando o modelo HuggingFace configurado
- Armazena no ChromaDB na coleção `jurisprudencia_br_v1`
- Processa em lotes para eficiência

**Tempo estimado**: Depende do volume de documentos (pode levar minutos a horas)

### Passo 3: Indexação do Código Penal

Em seguida, indexe o Código Penal brasileiro:

```bash
python create_db_cp.py
```

**Pré-requisito**: Certifique-se de que o arquivo `dados_sanitizados/codigo_penal/codigo_penal_estruturado.json` existe.

**O que este script faz**:
- Carrega o JSON estruturado do Código Penal
- Cria documentos por artigo com metadados completos
- Gera embeddings e armazena na coleção `legislacao_codigo_penal`
- Mais rápido que a jurisprudência (menos documentos)

### Passo 4: Executar a Interface Web (Streamlit)

Com os dados indexados, você pode usar a interface web:

1. **Inicie o servidor Streamlit**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Acesse no navegador**:
   - O Streamlit abrirá automaticamente em `http://localhost:8501`
   - Ou acesse manualmente: `http://localhost:8501`

3. **Use a interface**:
   - Digite perguntas sobre direito penal
   - Visualize fontes e scores
   - Explore o histórico de conversas

### Passo 5: Configurar e Executar o Bot WhatsApp (Opcional)

Para usar o bot WhatsApp, você precisa integrar com o **Twilio**.

#### 5.1: Configuração do Twilio

1. **Crie uma conta no Twilio**:
   - Acesse: https://www.twilio.com
   - Crie uma conta gratuita (inclui créditos para testes)

2. **Obtenha suas credenciais**:
   - No dashboard do Twilio, encontre:
     - `Account SID`
     - `Auth Token`
   - Configure um número WhatsApp (Sandbox ou produção)

3. **Atualize o `.env`**:
   ```env
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN=seu-auth-token-aqui
   TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886  # Seu número Twilio
   ```

#### 5.2: Instalação do ngrok (Para Desenvolvimento Local)

O Twilio precisa de um webhook público. Para desenvolvimento local, use **ngrok**:

1. **Instale o ngrok**:
   - Acesse: https://ngrok.com
   - Baixe e instale para seu sistema
   - Ou via package manager:
     ```bash
     # Linux (snap)
     snap install ngrok
     
     # Mac (Homebrew)
     brew install ngrok
     
     # Windows: baixe do site
     ```

2. **Crie uma conta ngrok** (gratuita):
   - Registre-se em https://dashboard.ngrok.com
   - Obtenha seu authtoken

3. **Configure o authtoken**:
   ```bash
   ngrok config add-authtoken seu-token-aqui
   ```

#### 5.3: Executar o Bot com ngrok

1. **Inicie o servidor Flask** (em um terminal):
   ```bash
   python whatssap_bot.py
   ```
   O servidor iniciará em `http://localhost:5050`

2. **Inicie o ngrok** (em outro terminal):
   ```bash
   ngrok http 5050
   ```

3. **Copie a URL pública do ngrok**:
   - Exemplo: `https://abc123.ngrok.io`
   - Use esta URL para configurar o webhook no Twilio

4. **Configure o webhook no Twilio**:
   - No dashboard do Twilio, vá em "Messaging" → "Settings" → "WhatsApp Sandbox"
   - Ou configure via API/Console
   - Defina o webhook como: `https://abc123.ngrok.io/webhook`

5. **Teste o bot**:
   - Envie uma mensagem para o número WhatsApp do Twilio
   - O bot deve responder automaticamente

**Importante**: 
- A URL do ngrok muda a cada execução (free tier)
- Para produção, use um servidor com URL fixa
- O bot responde em background para evitar timeout do Twilio

### Resumo do Workflow Completo

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate

# 2. Sanitizar PDFs (primeira vez)
python sanitaze.py

# 3. Indexar jurisprudência (primeira vez)
python create_db_jurisprudencia.py

# 4. Indexar Código Penal (primeira vez)
python create_db_cp.py

# 5. Executar Streamlit (qualquer momento)
streamlit run streamlit_app.py

# 6. Executar bot WhatsApp (opcional, em terminais separados)
# Terminal 1:
python whatssap_bot.py

# Terminal 2:
ngrok http 5050
```

### Verificação do Sistema

Após seguir todos os passos, teste o sistema:

```bash
# Teste direto via CLI
python rag_core.py "O que é legítima defesa?"
```

Se tudo estiver funcionando, você verá uma resposta com fontes citadas.

---

## 📁 Estrutura do Projeto

```
claiton-app/
├── streamlit_app.py              # Interface web principal
├── rag_core.py                   # Motor RAG e lógica de retrieval
├── whatssap_bot.py               # Bot WhatsApp
├── sanitaze.py                   # Sanitização de PDFs jurídicos
├── create_db_jurisprudencia.py  # Indexação de jurisprudência
├── create_db_cp.py               # Indexação do Código Penal
├── requirements.txt              # Dependências Python
├── README.md                     # Esta documentação
├── .env                          # Variáveis de ambiente (não versionado)
├── dados_sanitizados/           # Dados processados
│   ├── acordaos/                # Acórdãos estruturados (JSON)
│   ├── chunks/                   # Chunks para indexação
│   └── codigo_penal/            # Estrutura do Código Penal
└── vectordb/                    # Banco de dados vetorial
    └── chroma/                  # ChromaDB persistente
```

---

## 🛠️ Tecnologias, Algoritmos e Conceitos

### Inteligência Artificial e Machine Learning

#### 1. Retrieval Augmented Generation (RAG)

**RAG** é uma arquitetura híbrida que combina busca de informações com geração de texto:

- **Retrieval (Recuperação)**: Busca documentos relevantes em uma base de conhecimento
- **Augmentation (Aumento)**: Enriquece o contexto do LLM com documentos recuperados
- **Generation (Geração)**: Gera respostas baseadas no contexto aumentado

**Vantagens do RAG**:
- Reduz alucinações (respostas inventadas)
- Permite citações de fontes
- Atualização de conhecimento sem retreinar o modelo
- Especialização em domínios específicos (direito, neste caso)

#### 2. Embeddings Vetoriais

**Embeddings** são representações numéricas de texto em espaços vetoriais de alta dimensão:

- **Modelo utilizado**: `sentence-transformers` (baseado em arquitetura BERT/E5)
- **Dimensão**: Tipicamente 384 ou 768 dimensões
- **Princípio**: Textos semanticamente similares ficam próximos no espaço vetorial

**Como funciona**:
```
Texto → Modelo de Embedding → Vetor numérico (ex: [0.23, -0.45, 0.12, ...])
```

**Busca por Similaridade**:
- Queries e documentos são convertidos em vetores
- Cálculo de similaridade cosseno ou distância euclidiana
- Retorna documentos mais similares à query

#### 3. Vector Databases (Bancos de Dados Vetoriais)

**ChromaDB** é um banco de dados especializado em busca vetorial:

- **Armazenamento eficiente**: Otimizado para vetores de alta dimensão
- **Indexação**: Usa algoritmos como HNSW (Hierarchical Navigable Small World)
- **Busca rápida**: Encontra vetores similares em milhares/milhões de documentos
- **Persistência**: Dados armazenados localmente para reutilização

**Conceitos Chave**:
- **Coleções**: Agrupamento lógico de documentos relacionados
- **Metadados**: Informações estruturadas sobre cada documento
- **Similarity Search**: Busca por similaridade vetorial (não exata)

#### 4. Large Language Models (LLMs)

**Ollama** executa LLMs localmente:

- **Modelos suportados**: LLaMA, Mistral, CodeLlama, etc.
- **Inferência local**: Privacidade e controle total
- **Sem custos de API**: Não depende de serviços externos pagos
- **Customização**: Ajuste de temperatura, top_p, contexto

**Parâmetros Importantes**:
- **Temperature**: Controla aleatoriedade (0.0 = determinístico, 1.0 = criativo)
- **Top-P (Nucleus Sampling)**: Considera apenas tokens com probabilidade acumulada
- **Context Window**: Tamanho máximo do contexto (tokens)

#### 5. Processamento de Linguagem Natural (NLP)

Técnicas NLP aplicadas:

- **Text Chunking**: Divisão de documentos longos em pedaços menores
- **Text Sanitization**: Remoção de ruídos (copyright, headers, footers)
- **Metadata Extraction**: Extração de informações estruturadas via regex/NER
- **Semantic Search**: Busca por significado, não palavras exatas

### Arquitetura e Padrões de Design

#### 1. Dual Retrieval System

Sistema que busca em **duas coleções** simultaneamente:

- **Jurisprudência**: Precedentes e decisões judiciais
- **Legislação**: Artigos do Código Penal

**Vantagens**:
- Respostas mais completas (lei + jurisprudência)
- Flexibilidade para balancear fontes
- Especialização por tipo de documento

#### 2. Prompt Engineering

Técnica de construção de prompts para guiar o LLM:

- **System Instructions**: Diretrizes gerais de comportamento
- **Context Formatting**: Estruturação do contexto recuperado
- **Output Formatting**: Especificação do formato de resposta desejado

**Exemplo no projeto**:
```
[SISTEMA]
Você é um assistente jurídico especializado...

[PERGUNTA DO USUÁRIO]
{question}

[CONTEXTOS RECUPERADOS]
{contexts}

[INSTRUÇÕES DE SAÍDA]
- Responda em português...
- Cite as fontes...
```

#### 3. Asynchronous Processing

No bot WhatsApp, processamento assíncrono:

- **Threading**: Respostas em background para evitar timeout
- **TwiML**: Resposta imediata ao Twilio
- **API Calls**: Envio da resposta completa via API após processamento

### Tecnologias e Bibliotecas

#### Core Technologies

- **Python 3.8+**: Linguagem principal do projeto
  - Tipagem opcional, bibliotecas ricas
  - Suporte a async/await

- **LangChain**: Framework para aplicações LLM
  - Abstrações para RAG
  - Integração com múltiplos vetores DBs
  - Gerenciamento de documentos e chains

- **ChromaDB**: Banco de dados vetorial
  - Open-source e local
  - API Python intuitiva
  - Persistência em disco

- **HuggingFace Transformers**: Modelos de embeddings
  - Biblioteca padrão para NLP
  - Modelos pré-treinados em múltiplas línguas
  - Suporte a GPU/CPU

- **Ollama**: Runtime para LLMs
  - Execução local de modelos
  - API REST simples
  - Otimizações para CPU/GPU

#### Interface e Integrações

- **Streamlit**: Framework web para Python
  - Desenvolvimento rápido de interfaces
  - Componentes pré-construídos (chat, forms, etc.)
  - Hot-reload para desenvolvimento

- **Twilio**: API para WhatsApp
  - Integração oficial com WhatsApp Business
  - Webhooks para mensagens
  - Sandbox para desenvolvimento

- **Flask**: Microframework web
  - Leve e flexível
  - Ideal para APIs e webhooks
  - Roteamento simples

#### Processamento de Dados

- **PyPDF2**: Extração de texto de PDFs
  - Parser de PDFs
  - Extração de texto e metadados
  - Suporte a múltiplas páginas

- **sentence-transformers**: Geração de embeddings
  - Modelos otimizados para embeddings
  - Batch processing
  - Suporte a múltiplas línguas

- **NumPy**: Computação numérica
  - Operações com vetores e matrizes
  - Base para embeddings

- **Pandas**: Manipulação de dados
  - Estruturas de dados tabulares
  - Processamento de datasets

### Algoritmos de Busca e Similaridade

#### 1. Cosine Similarity

Medida de similaridade entre vetores:

```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

- Valor entre -1 e 1
- 1 = idênticos, 0 = ortogonais, -1 = opostos
- Usado para comparar embeddings de texto

#### 2. HNSW (Hierarchical Navigable Small World)

Algoritmo de indexação para busca vetorial:

- **Grafos hierárquicos**: Múltiplas camadas de conexões
- **Busca eficiente**: O(log N) para encontrar vizinhos mais próximos
- **Aproximação**: Trade-off entre velocidade e precisão

#### 3. Text Chunking Strategies

Estratégias de divisão de texto:

- **Fixed-size chunks**: Divisão em tamanhos fixos (palavras/tokens)
- **Overlapping windows**: Sobreposição para preservar contexto
- **Semantic chunking**: Divisão baseada em significado (não implementado, mas possível)

### Conceitos de Machine Learning Aplicados

#### 1. Transfer Learning

- Modelos pré-treinados (BERT, E5) adaptados para português
- Fine-tuning não necessário (zero-shot)
- Aproveitamento de conhecimento de modelos grandes

#### 2. Zero-Shot Learning

- Modelos generalizam para tarefas não vistas durante treinamento
- Funciona com prompts bem estruturados
- Sem necessidade de dados de treino específicos

#### 3. Semantic Understanding

- Compreensão de significado, não apenas palavras
- "Homicídio" e "assassinato" são tratados como similares
- Busca por conceitos, não termos exatos

### Pipeline de Dados

```
PDF → Extração → Sanitização → Chunking → Embedding → Vector DB
                                                        ↓
Query → Embedding → Similarity Search → Retrieval → Context
                                                        ↓
Context + Query → LLM → Generated Response
```

### Métricas e Avaliação

- **Relevance Score**: Similaridade entre query e documento
- **Retrieval Quality**: Precisão dos documentos recuperados
- **Response Quality**: Avaliação subjetiva das respostas geradas
- **Latency**: Tempo de resposta do sistema completo

---

## 🤝 Contribuição

Este projeto foi desenvolvido como TCC. Para contribuições:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como Trabalho de Conclusão de Curso (TCC) - 2025.

---

## ⚠️ Avisos Importantes

- **Este sistema é uma ferramenta de apoio**: Não substitui consulta jurídica profissional
- **Validação de informações**: Sempre verifique as fontes citadas
- **Uso responsável**: As respostas são baseadas em documentos indexados e podem não estar atualizadas

---

## 📞 Contato

Para dúvidas ou sugestões sobre o projeto, entre em contato através dos canais apropriados.

---

**Desenvolvido com ❤️ para facilitar o acesso à informação jurídica brasileira**

