"""Testes do limitador de tarefas em voo (backpressure do worker)."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.core.task_limiter import LimitedExecutor


def test_submit_bloqueia_quando_limite_atingido():
    with ThreadPoolExecutor(max_workers=2) as pool:
        limitado = LimitedExecutor(pool, max_in_flight=2)
        portao = threading.Event()

        def tarefa():
            portao.wait(timeout=5)

        limitado.submit(tarefa)
        limitado.submit(tarefa)

        terceiro_submetido = threading.Event()

        def submeter_terceiro():
            limitado.submit(tarefa)
            terceiro_submetido.set()

        t = threading.Thread(target=submeter_terceiro, daemon=True)
        t.start()

        # Com o limite cheio, o terceiro submit deve ficar bloqueado
        assert not terceiro_submetido.wait(timeout=0.3)

        portao.set()  # libera as tarefas em voo
        assert terceiro_submetido.wait(timeout=5)


def test_limite_e_liberado_apos_conclusao():
    with ThreadPoolExecutor(max_workers=1) as pool:
        limitado = LimitedExecutor(pool, max_in_flight=1)
        assert limitado.submit(lambda: 42).result(timeout=5) == 42
        assert limitado.submit(lambda: 43).result(timeout=5) == 43


def test_limite_e_liberado_quando_submit_falha():
    """Se o executor interno lança no submit, o semáforo deve ser devolvido —
    senão o worker travaria para sempre após max_in_flight falhas."""

    class ExecutorQueFalha:
        def submit(self, fn, *args, **kwargs):
            raise RuntimeError("pool encerrado")

    limitado = LimitedExecutor(ExecutorQueFalha(), max_in_flight=1)

    with pytest.raises(RuntimeError):
        limitado.submit(lambda: None)

    # Se o semáforo vazou, esta segunda chamada bloquearia em vez de lançar
    segunda_chamada = threading.Event()

    def tentar_de_novo():
        try:
            limitado.submit(lambda: None)
        except RuntimeError:
            pass
        segunda_chamada.set()

    t = threading.Thread(target=tentar_de_novo, daemon=True)
    t.start()
    assert segunda_chamada.wait(timeout=1), "semáforo não foi liberado após falha no submit"
