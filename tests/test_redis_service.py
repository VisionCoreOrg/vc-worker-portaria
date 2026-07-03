"""Testes do serviço Redis do worker (fila confiável com ack)."""

import json
from unittest.mock import MagicMock

from src.config import REDIS_QUEUE
from src.services.redis_service import (
    FILA_PROCESSAMENTO,
    aguardar_evento,
    confirmar_evento,
    recuperar_eventos_orfaos,
)


def test_aguardar_evento_move_para_processing_com_blmove():
    """O consumo usa BLMOVE cauda→cabeça para reter o evento em processing."""
    cliente = MagicMock()
    payload = {"path": "dataset/img.jpg", "camera_id": "cam_01", "timestamp": "2026-07-02T00:00:00+00:00"}
    bruto = json.dumps(payload)
    cliente.blmove.return_value = bruto

    resultado = aguardar_evento(cliente, timeout=5)

    cliente.blmove.assert_called_once_with(
        REDIS_QUEUE, FILA_PROCESSAMENTO, 5, src="RIGHT", dest="LEFT"
    )
    assert resultado == (payload, bruto)


def test_aguardar_evento_timeout_retorna_none():
    cliente = MagicMock()
    cliente.blmove.return_value = None
    assert aguardar_evento(cliente, timeout=1) is None


def test_aguardar_evento_json_invalido_descarta_e_remove_de_processing():
    cliente = MagicMock()
    cliente.blmove.return_value = "{nao-e-json"

    assert aguardar_evento(cliente, timeout=1) is None
    cliente.lrem.assert_called_once_with(FILA_PROCESSAMENTO, 1, "{nao-e-json")


def test_confirmar_evento_remove_da_lista_de_processamento():
    cliente = MagicMock()
    confirmar_evento(cliente, '{"path": "x"}')
    cliente.lrem.assert_called_once_with(FILA_PROCESSAMENTO, 1, '{"path": "x"}')


def test_recuperar_eventos_orfaos_devolve_ate_esvaziar():
    cliente = MagicMock()
    cliente.lmove.side_effect = ['{"path": "a"}', '{"path": "b"}', None]

    assert recuperar_eventos_orfaos(cliente) == 2
    cliente.lmove.assert_called_with(
        FILA_PROCESSAMENTO, REDIS_QUEUE, src="LEFT", dest="RIGHT"
    )
