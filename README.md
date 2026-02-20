# 🚗 Worker 01 - Portaria (Real-Time OCR)

Este repositório contém o microsserviço responsável pelo processamento de Visão Computacional e OCR das câmeras da guarita de entrada do estacionamento.

Ele atua de forma assíncrona consumindo filas de alta prioridade, garantindo processamento ágil para manter a fluidez na catraca física.

## 🏗️ Arquitetura e Fluxo
  
A solução foi desenhada para não bloquear a API Core transacional. O fluxo funciona da seguinte maneira:

1. Uma imagem da placa é capturada na guarita.

2. O payload (imagem/caminho) é publicado em uma fila do **Redis**.

3. O **Celery** consome essa fila instantaneamente.

4. O script de OpenCV/Tesseract processa a imagem, extrai a string da placa e calcula o grau de confiança.

5. O resultado, juntamente com logs de tempo em milissegundos, é devolvido para a API Core.

## 🛠️ Stack Tecnológica
  
* **Linguagem:** Python 3.15.0a6-slim

* **Mensageria:** Redis

* **Task Queue:** Celery

* **Visão Computacional:** OpenCV + Tesseract / EasyOCR

* **Infraestrutura:** Docker & Docker Compose

## 🚀 Como Rodar o Ambiente Local

A infraestrutura está totalmente conteinerizada para garantir paridade com o ambiente de produção. Não é necessário instalar bibliotecas de C/C++ do OpenCV na sua máquina hospedeira.

### Pré-requisitos

* Docker e Docker Compose instalados.

### Passos de Execução

1. Clone o repositório.
2. Crie uma cópia do arquivo `.env.example` e renomeie para `.env` (preencha as senhas se necessário).
3. Suba o banco de dados e o Redis em background com o comando:
   ```bash
   docker-compose up -d

```
4. O PostgreSQL estará rodando na porta 5432 e o Redis na porta 6379.