"""Testes do limitador de tarefas em voo (backpressure do worker)."""

import threading
from concurrent.futures import ThreadPoolExecutor

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
