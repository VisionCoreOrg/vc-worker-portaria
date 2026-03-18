import requests
from src.config import API_URL

def enviar_para_api(payload):
    """Envia os dados da placa reconhecida para a API Core (FastAPI)."""
    try:
        resposta = requests.post(API_URL, json=payload, timeout=5)
        if resposta.status_code in [200, 201]:
            print(f"[API] Registro enviado com sucesso! Resposta: {resposta.text}")
            return True
        else:
            print(f"[API] Erro na API Core: {resposta.status_code} - {resposta.text}")
            return False
    except Exception as e:
        print(f"[API] 🚨 API Core offline ou inatingível: {e}")
        return False