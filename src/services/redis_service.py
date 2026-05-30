from typing import Optional
import json
import time
import redis
from src.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_QUEUE
from src.core.logger import configurar_logger

logger = configurar_logger("RedisService")


def criar_cliente_redis() -> redis.Redis:
    """Cria e retorna um cliente Redis autenticado com decode_responses=True."""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )


def aguardar_evento(cliente: redis.Redis, timeout: int = 5) -> Optional[dict]:
    """
    Bloqueia esperando por uma mensagem na fila Redis (BLPOP).
    Retorna o payload desserializado ou None se não houver mensagem no timeout.
    """
    try:
        resultado = cliente.blpop(REDIS_QUEUE, timeout=timeout)
        if resultado is None:
            return None
        
        _, mensagem_json = resultado
        return json.loads(mensagem_json)
    except json.JSONDecodeError as e:
        logger.warning(f"Mensagem inválida ignorada (JSON inválido): {e}")
        return None
    except redis.exceptions.RedisError as e:
        logger.error(f"Erro ao interagir com o Redis (BLPOP): {e}")
        time.sleep(1.0)  # Pequeno delay antes de reatar para evitar loop rápido de erros
        return None


def conectar_com_retry(tentativas: int = 10, espera: int = 3) -> redis.Redis:
    """
    Tenta conectar ao Redis com retry simples.
    Útil para aguardar o redis_broker subir antes do worker.
    """
    for i in range(1, tentativas + 1):
        try:
            cliente = criar_cliente_redis()
            cliente.ping()
            logger.info(f"Conectado com sucesso ao broker Redis em {REDIS_HOST}:{REDIS_PORT}")
            return cliente
        except redis.exceptions.ConnectionError as e:
            logger.warning(f"Tentativa {i}/{tentativas} — Redis indisponível em {REDIS_HOST}:{REDIS_PORT}: {e}")
            if i < tentativas:
                time.sleep(espera)
    
    logger.critical(f"Não foi possível conectar ao Redis após {tentativas} tentativas.")
    raise RuntimeError(f"[REDIS] Não foi possível conectar após {tentativas} tentativas.")
