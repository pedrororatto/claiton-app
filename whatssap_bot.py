#!/usr/bin/env python3
"""
Bot WhatsApp para RAG Jurídico via Twilio
"""
import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from rag_core import answer_question
from threading import Thread
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

# Credenciais Twilio (via .env)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.values.get("Body", "").strip()
    from_number = request.values.get("From", "")

    print(f"[RECEBIDO] {from_number}: {incoming_msg}")

    # Validações
    if not incoming_msg or len(incoming_msg) < 10:
        return respond("Por favor, envie uma pergunta mais detalhada.")

    # Responder imediatamente ao Twilio
    response = MessagingResponse()
    response.message("⏳ Processando sua pergunta... Aguarde alguns segundos.")

    # Processar em background e enviar depois
    Thread(target=process_and_send, args=(incoming_msg, from_number)).start()
    return str(response)


def process_and_send(question, to_number):
    """Processa RAG e envia resposta via Twilio API"""
    try:
        print(f"[BACKGROUND] Processando: {question}")
        # Limitar resposta a 1200 chars para deixar espaço para header e fontes (total max 1600)
        resposta, fontes = answer_question(question, max_response_length=1200)
        
        # Formatar mensagem
        mensagem = f"📋 *Resposta:*\n{resposta}\n\n"
        if fontes:
            mensagem += f"📚 *Fontes:*\n"
            for i, fonte in enumerate(fontes[:3], 1):
                titulo = fonte.get("titulo", "N/A")
                origem = fonte.get("origem", "N/A")
                mensagem += f"{i}. {titulo} ({origem})\n"

        # Garantir que não exceda 1600 caracteres (limite do Twilio)
        if len(mensagem) > 1600:
            # Truncar mensagem completa se necessário
            mensagem = mensagem[:1597] + "..."

        print(f"[DEBUG] Tamanho da mensagem: {len(mensagem)} caracteres")

        # Enviar via Twilio API
        client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        print("meu cliente: ", client)
        message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=mensagem,
            to=to_number
        )
        print("mensagem enviada: ", message)

        print(f"[ENVIADO] SID: {message.sid}")

    except Exception as e:
        print(f"[ERRO BACKGROUND] {e}")
        import traceback
        traceback.print_exc()

def respond(message):
    """Cria resposta TwiML"""
    response = MessagingResponse()
    response.message(message)
    return str(response)


@app.route("/status", methods=["GET"])
def status():
    """Health check"""
    return {"status": "online", "service": "RAG Jurídico WhatsApp Bot"}


if __name__ == "__main__":
    # Para desenvolvimento local com ngrok
    app.run(host="0.0.0.0", port=5050, debug=True)