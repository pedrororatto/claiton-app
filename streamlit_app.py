#!/usr/bin/env python3
"""
Interface Web para RAG Jurídico usando Streamlit + streamlit-chat
Versão com suporte a tema escuro e texto integral
"""
import streamlit as st
from streamlit_chat import message
import time
from datetime import datetime
from rag_core import answer_question

# Configuração da página
st.set_page_config(
    page_title="CLAITON - Assistente Jurídico",
    page_icon="⚖️",
    layout="wide"
)

# CSS customizado com suporte a tema escuro
st.markdown("""
<style>
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        color: var(--text-color);
    }

    /* Input maior */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }

    /* Expander customizado */
    .streamlit-expanderHeader {
        font-weight: 600;
    }

    /* Melhorar contraste dos botões */
    .stButton > button {
        font-weight: 500;
    }

    /* Texto integral com scroll */
    .texto-integral {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(128, 128, 128, 0.1);
        font-size: 0.9rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Mensagem de boas-vindas
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Olá! Sou seu assistente jurídico. Posso ajudá-lo com consultas sobre jurisprudência e legislação brasileira. Como posso ajudar?",
        "fontes": []
    })

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# Sidebar
with st.sidebar:
    st.markdown("## ⚖️ CLAITON")
    st.markdown("---")

    st.markdown("### 📊 Estatísticas")
    st.metric("Consultas", st.session_state.total_queries)
    st.metric("Mensagens", len(st.session_state.messages))

    st.markdown("---")

    st.markdown("### ⚙️ Configurações")
    max_sources = st.slider("Fontes a exibir", 1, 10, 3)
    show_scores = st.checkbox("Mostrar scores", value=False)
    show_texto_integral = st.checkbox("Mostrar texto integral", value=False)

    st.markdown("---")

    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]  # Manter boas-vindas
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("---")

    st.markdown("### 💡 Exemplos")
    if st.button("O que é legítima defesa?", use_container_width=True):
        st.session_state.example_query = "O que é legítima defesa?"
    if st.button("Pena para furto qualificado", use_container_width=True):
        st.session_state.example_query = "Qual a pena para furto qualificado?"
    if st.button("Crimes contra a honra", use_container_width=True):
        st.session_state.example_query = "Jurisprudência sobre crimes contra a honra"

    st.markdown("---")
    st.caption("💻 Desenvolvido para TCC - 2025")

# Header
st.markdown('<div class="main-header">⚖️ CLAITON - Seu Assistente Jurídico Inteligente</div>', unsafe_allow_html=True)

st.info("⚠️ **Aviso:** Este sistema é uma ferramenta de apoio. Não substitui consulta jurídica profissional.")

# Container de chat
chat_container = st.container()

with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        is_user = msg["role"] == "user"

        # Exibir mensagem
        message(
            msg["content"],
            is_user=is_user,
            key=f"msg_{i}",
            avatar_style="avataaars" if is_user else "bottts"
        )

        # Exibir fontes (apenas para mensagens do assistente)
        if not is_user and msg.get("fontes"):
            num_fontes = len(msg["fontes"])

            with st.expander(f"📚 Ver {num_fontes} fonte(s) consultada(s)", expanded=False):
                for idx, fonte in enumerate(msg["fontes"][:max_sources], 1):
                    titulo = fonte.get("titulo", "N/A")
                    origem = fonte.get("origem", "N/A")
                    score = fonte.get("score", 0)
                    text = fonte.get("text", fonte.get("text", "Texto não disponível"))

                    # Header da fonte
                    st.markdown(f"**[{idx}] {titulo}**")
                    st.caption(f"📂 Origem: {origem}")

                    # Score (se habilitado)
                    if show_scores:
                        st.caption(f"🎯 Score: {score:.4f}")

                    # Texto integral (se habilitado)
                    if show_texto_integral:
                        with st.expander("📄 Ver decisão completa", expanded=False):
                            st.markdown(f'<div class="texto-integral">{text}</div>', unsafe_allow_html=True)

                    # Separador entre fontes
                    if idx < min(num_fontes, max_sources):
                        st.divider()

# Input do usuário
st.markdown("---")

# Verificar se há exemplo selecionado
default_value = ""
if "example_query" in st.session_state:
    default_value = st.session_state.example_query
    del st.session_state.example_query

user_input = st.text_input(
    "Digite sua pergunta:",
    value=default_value,
    placeholder="Ex: O que caracteriza legítima defesa no direito penal?",
    key="user_input"
)

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    send_button = st.button("📤 Enviar", use_container_width=True)

with col2:
    clear_input = st.button("🔄 Limpar", use_container_width=True)

# Processar envio
if send_button and user_input:
    if len(user_input.strip()) < 10:
        st.error("⚠️ Pergunta muito curta. Seja mais específico.")
    else:
        # Adicionar pergunta do usuário
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "fontes": []
        })

        st.session_state.total_queries += 1

        # Processar resposta
        with st.spinner("🔍 Analisando documentos e gerando resposta..."):
            try:
                start_time = time.time()
                resposta, fontes = answer_question(user_input)
                elapsed_time = time.time() - start_time

                # Adicionar resposta
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resposta,
                    "fontes": fontes
                })

                st.success(f"✅ Resposta gerada em {elapsed_time:.1f}s")
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Desculpe, ocorreu um erro. Tente novamente.",
                    "fontes": []
                })
                st.rerun()

if clear_input:
    st.rerun()