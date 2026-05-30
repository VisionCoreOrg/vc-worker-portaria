# ADR 0002: Processamento Paralelo Concorrente de Eventos via `ThreadPoolExecutor`

* **Data:** 2026-05-30
* **Status:** Aprovado

## Contexto e Problema

O worker `vc-worker-portaria` operava em modo estritamente linear, síncrono e thread-única. O loop de consumo realizava em sequência:
1. `BLPOP` na fila Redis (baixo custo).
2. Download de imagem original via S3 (I/O bloqueante).
3. Inferência de Visão Computacional YOLOv8 (CPU/GPU-bound).
4. Processamento e inferência OCR EasyOCR (CPU/GPU-bound).
5. Upload de mídias de crops via S3 (I/O bloqueante).
6. Requisição HTTP POST para a API Core (I/O bloqueante).

Se a rede com a API ou com o S3 sofresse qualquer tipo de degradação de performance ou timeout, a thread principal do worker ficava completamente paralisada aguardando o timeout de soquete. Como consequência, novos eventos de placas no portão de entrada acumulavam-se na fila Redis, inviabilizando a operação de reconhecimento de placas em tempo real (LPR) do estacionamento.

## Decisão

Implementamos um padrão de concorrência baseado em **`ThreadPoolExecutor`** (módulo nativo `concurrent.futures`) de 4 threads no bootstrapping do script `src/main.py`:

1. A thread principal do worker é reservada unicamente para escutar mensagens do Redis de forma linear via `BLPOP` (uma operação muito rápida e de baixo overhead).
2. Assim que um evento de frame de câmera é retirado da fila Redis, a tarefa de orquestração do caso de uso (`use_case.executar(evento)`) é imediatamente delegada e submetida a uma thread livre do pool concorrente.
3. As threads realizam downloads, inferências matemáticas pesadas de IA, uploads e chamadas HTTP de forma paralela concorrente.

## Justificativa Técnica (Por que threads e não `asyncio`?)

As bibliotecas pesadas de machine learning utilizadas pelo worker (como ONNX Runtime escrito em C++ nativo e o PyTorch por trás do EasyOCR) realizam suas computações pesadas fora do interpretador padrão do Python. Elas **liberam o Global Interpreter Lock (GIL)** do Python durante as fases intensivas de inferência matemática. 

Portanto, o uso de múltiplas threads de sistema é altamente eficiente para este cenário, proporcionando paralelismo real de computação além de paralelizar o I/O de rede. Reescrever a aplicação inteira para o paradigma de corrotinas assíncronas (`asyncio`) exigiria substituir todas as bibliotecas padrão por variantes assíncronas (`aioboto3`, `httpx`, `redis-py` assíncrono), aumentando drasticamente a complexidade do código, sem resolver o problema de CPU-bound das inferências de IA (que continuariam necessitando ser empurradas para pools de threads separados para não congelar o loop de eventos assíncronos).

## Consequências

### Positivas
* **Throughput Aumentado:** O worker agora processa múltiplos frames de câmera de forma concorrente em paralelo. Quedas de performance ou latências na API Core não congelam o consumo de novos eventos da guarita.
* **Baixo Impacto no Código:** A refatoração concorrente exigiu apenas cerca de 10 linhas de código no arquivo `main.py`, sem a necessidade de reescrever as assinaturas e comportamentos de I/O em todos os outros arquivos de serviços para `async/await`.

### Negativas
* **Maior Consumo de Recursos:** A execução em paralelo exige maior concorrência de CPU, memória RAM e recursos de hardware concorrendo por threads de execução da GPU (se ativada). O limite padrão foi fixado em 4 threads simultâneas para equilibrar consumo e velocidade de resposta em CPU padrão de desenvolvimento.
