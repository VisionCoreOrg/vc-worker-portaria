import requests
from src.config import API_URL

def enviar_para_api(payload):
    """Dispara os dados da placa para o seu backend NestJS"""
    try:
        resposta = requests.post(API_URL, json=payload, timeout=5)
        if resposta.status_code in [200, 201]:
            print(f"Enviado com sucesso! NestJS respondeu: {resposta.text}")
            return True
        else:
            print(f"Erro no FastApi: {resposta.status_code} - {resposta.text}")
            return False
    except Exception as e:
        print(f"🚨 Backend offline ou inatingível: {e}")
        return False