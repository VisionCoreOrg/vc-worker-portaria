"""Limitador de tarefas em voo para o pool de threads do worker.

Sem este limite, o loop principal retira eventos do Redis mais rápido do
que o pool processa: a fila interna do ThreadPoolExecutor cresce sem
limite na memória e os eventos (já removidos do broker) se perdem em um
crash. O submit bloqueante propaga a pressão de volta para o BRPOP.
"""

import threading
from concurrent.futures import Executor, Future
from typing import Any, Callable


class LimitedExecutor:
    """Envolve um Executor limitando o número de tarefas em voo.

    ``submit`` bloqueia quando ``max_in_flight`` tarefas ainda não
    terminaram, criando backpressure natural no consumo da fila.
    """

    def __init__(self, executor: Executor, max_in_flight: int):
        self._executor = executor
        self._semaforo = threading.BoundedSemaphore(max_in_flight)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submete a tarefa, bloqueando se o limite de tarefas em voo foi atingido."""
        self._semaforo.acquire()
        try:
            future = self._executor.submit(fn, *args, **kwargs)
        except BaseException:
            self._semaforo.release()
            raise
        future.add_done_callback(lambda _f: self._semaforo.release())
        return future
