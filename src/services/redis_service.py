from typing import Optional, Tuple
import json
import time
import redis
from src.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_QUEUE
from src.core.logger import configurar_logger

logger = configurar_logger("RedisService")

# Lista auxiliar onde o evento fica retido enquanto é processado (padrão de
# fila confiável: BLMOVE fila -> processing no consumo, LREM no ack).
FILA_PROCESSAMENTO = f"{REDIS_QUEUE}:processing"


def criar_cliente_redis() -> redis.Redis:
    """Cria e retorna um cliente Redis autenticado com decode_responses=True."""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )


def aguardar_evento(cliente: redis.Redis, timeout: int = 5) -> Optional[Tuple[dict, str]]:
    """
    Bloqueia esperando por uma mensagem na fila Redis (BLMOVE).
    Move o evento da cauda da fila para a cabeça de FILA_PROCESSAMENTO, onde
    fica retido até o ack (confirmar_evento) — um crash não perde o evento.
    Com o produtor publicando via LPUSH na cabeça, o consumo pela cauda
    mantém a fila FIFO.
    Retorna (payload desserializado, mensagem bruta) ou None se não houver
    mensagem no timeout. A mensagem bruta é necessária para o ack.
    """
    mensagem_json = None
    try:
        mensagem_json = cliente.blmove(
            REDIS_QUEUE, FILA_PROCESSAMENTO, timeout, src="RIGHT", dest="LEFT"
        )
        if mensagem_json is None:
            return None
        return json.loads(mensagem_json), mensagem_json
    except json.JSONDecodeError as e:
        # Sem este LREM a mensagem inválida ficaria órfã em processing e
        # voltaria para a fila a cada restart do worker.
        cliente.lrem(FILA_PROCESSAMENTO, 1, mensagem_json)
        logger.warning(f"Mensagem inválida descartada (JSON inválido): {e}")
        return None
    except redis.exceptions.RedisError as e:
        logger.error(f"Erro ao interagir com o Redis (BLMOVE): {e}")
        time.sleep(1.0)  # Pequeno delay antes de reatar para evitar loop rápido de erros
        return None


def confirmar_evento(cliente: redis.Redis, mensagem_bruta: str) -> None:
    """Ack: remove da lista de processamento o evento já processado."""
    try:
        cliente.lrem(FILA_PROCESSAMENTO, 1, mensagem_bruta)
    except redis.exceptions.RedisError as e:
        logger.error(f"Erro ao confirmar evento processado (LREM): {e}")


def recuperar_eventos_orfaos(cliente: redis.Redis) -> int:
    """
    Devolve à fila principal eventos retidos em processamento (sobras de um
    crash). Único worker: no startup, tudo que está na lista é órfão.
    Move da cabeça de processing para a cauda da fila — assim os órfãos mais
    antigos são consumidos primeiro e o FIFO global é preservado.
    Retorna o número de eventos devolvidos.
    """
    devolvidos = 0
    while cliente.lmove(FILA_PROCESSAMENTO, REDIS_QUEUE, src="LEFT", dest="RIGHT") is not None:
        devolvidos += 1
    if devolvidos:
        logger.warning(f"{devolvidos} evento(s) órfão(s) devolvido(s) à fila após restart.")
    return devolvidos


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
