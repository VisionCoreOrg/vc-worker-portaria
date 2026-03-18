import json
import time
import redis
from src.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_QUEUE


def criar_cliente_redis() -> redis.Redis:
    """Cria e retorna um cliente Redis autenticado."""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )


def aguardar_evento(cliente: redis.Redis, timeout: int = 5) -> dict | None:
    """
    Bloqueia esperando por uma mensagem na fila Redis (BLPOP).
    Retorna o payload desserializado ou None se não houver mensagem no timeout.
    """
    resultado = cliente.blpop(REDIS_QUEUE, timeout=timeout)
    if resultado is None:
        return None
    _, mensagem_json = resultado
    try:
        return json.loads(mensagem_json)
    except json.JSONDecodeError as e:
        print(f"[REDIS] Mensagem inválida ignorada: {e} | Raw: {mensagem_json}")
        return None


def conectar_com_retry(tentativas: int = 10, espera: int = 3) -> redis.Redis:
    """
    Tenta conectar ao Redis com retry exponencial simples.
    Útil para aguardar o redis_broker subir antes do worker.
    """
    for i in range(1, tentativas + 1):
        try:
            cliente = criar_cliente_redis()
            cliente.ping()
            print(f"[REDIS] Conectado com sucesso ao broker em {REDIS_HOST}:{REDIS_PORT}")
            return cliente
        except redis.exceptions.ConnectionError as e:
            print(f"[REDIS] Tentativa {i}/{tentativas} — Redis indisponível: {e}")
            if i < tentativas:
                time.sleep(espera)
    raise RuntimeError(f"[REDIS] Não foi possível conectar após {tentativas} tentativas.")
