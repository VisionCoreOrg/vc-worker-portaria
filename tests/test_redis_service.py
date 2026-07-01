"""Testes do serviço Redis do worker (consumo FIFO da fila de eventos)."""

import json
from unittest.mock import MagicMock

from src.config import REDIS_QUEUE
from src.services.redis_service import aguardar_evento


def test_aguardar_evento_consome_fifo_com_brpop():
    """O consumo deve usar BRPOP (cauda) para formar FIFO com o LPUSH do produtor."""
    cliente = MagicMock()
    payload = {"path": "dataset/img.jpg", "camera_id": "cam_01", "timestamp": "2026-07-01T00:00:00+00:00"}
    cliente.brpop.return_value = (REDIS_QUEUE, json.dumps(payload))

    evento = aguardar_evento(cliente, timeout=5)

    cliente.brpop.assert_called_once_with(REDIS_QUEUE, timeout=5)
    cliente.blpop.assert_not_called()
    assert evento == payload


def test_aguardar_evento_timeout_retorna_none():
    cliente = MagicMock()
    cliente.brpop.return_value = None
    assert aguardar_evento(cliente, timeout=1) is None


def test_aguardar_evento_json_invalido_retorna_none():
    cliente = MagicMock()
    cliente.brpop.return_value = (REDIS_QUEUE, "{nao-e-json")
    assert aguardar_evento(cliente, timeout=1) is None
