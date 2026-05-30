import requests
from src.core.logger import configurar_logger

logger = configurar_logger("FastAPIClient")

class FastAPIClient:
    """
    Implementa o APIClient conectando com a API Core FastAPI.
    Conforme com o Protocol 'APIClient'.
    """

    def __init__(self, api_url: str, timeout: float = 5.0):
        self.api_url = api_url
        self.timeout = timeout
        logger.info(f"Cliente API Core configurado no endpoint: {self.api_url}")

    def registrar_passagem(self, payload: dict) -> bool:
        """
        Envia os dados da passagem do veículo para o endpoint da API Core.
        """
        try:
            resposta = requests.post(self.api_url, json=payload, timeout=self.timeout)
            if resposta.status_code in [200, 201]:
                logger.info(f"Registro enviado com sucesso! Status: {resposta.status_code}")
                return True
            else:
                logger.error(
                    f"Erro de resposta na API Core. Status: {resposta.status_code} | "
                    f"Mensagem: {resposta.text.strip()}"
                )
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Falha de rede ao conectar com a API Core em '{self.api_url}': {e}")
            return False
        except Exception as e:
            logger.error(f"Erro inesperado no envio dos dados para a API: {e}")
            return False