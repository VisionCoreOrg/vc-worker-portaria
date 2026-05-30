# ADR 0001: Desacoplamento de IA/OCR e Inversão de Dependências via `typing.Protocol`

* **Data:** 2026-05-30
* **Status:** Aprovado

## Contexto e Problema

O microsserviço `vc-worker-portaria` operava de forma acoplada. O script orquestrador principal (`main.py`) importava e instanciava diretamente classes concretas de infraestrutura e bibliotecas de terceiros, como o `easyocr.Reader`, e dependia diretamente do tipo `ort.InferenceSession` do ONNX Runtime. 

Essa forte dependência direta violava o **Dependency Inversion Principle (DIP)** e o **Single Responsibility Principle (SRP)**. Além disso:
1. Dificultava a testabilidade unitária, pois era impossível isolar o loop de eventos sem inicializar os pesos dos modelos pesados de Machine Learning na CPU ou GPU.
2. Dificultava a troca ou evolução de frameworks de IA (ex: substituir EasyOCR pelo Tesseract ou por uma API de nuvem como Google Cloud Vision) sem a necessidade de reescrever o fluxo de negócio do worker.
3. Regras de negócio brasileiras de validação e formatação de placas estavam acopladas no mesmo arquivo que lidava com processamento de imagem OpenCV.

## Decisão

Implementamos os princípios de **Clean Architecture** (Arquitetura Limpa) por meio das seguintes alterações estruturais:

1. **Criação de Contratos (Interfaces):** Definimos protocolos em `src/core/interfaces.py` usando `typing.Protocol` do Python para representar de forma abstrata as dependências de rede e de machine learning:
   * `Detector` (YOLO)
   * `OCRReader` (EasyOCR)
   * `StorageRepository` (MinIO/S3)
   * `APIClient` (FastAPI)
2. **Caso de Uso Central:** Centralizamos a regra de orquestração na classe `ProcessarEventoUseCase` em `src/core/use_cases.py`. Esse caso de uso é 100% puro e depende estritamente das abstrações de interfaces declaradas no Core, sendo livre de imports de `boto3`, `requests` ou `easyocr`.
3. **Isolamento de Utilidades:** Modularizamos as manipulações matemáticas de imagens OpenCV em `src/utils/image_utils.py` (letterbox, NMS, grayscaling, Otsu thresholding) e a higienização de string / regras Mercosul brasileiras em `src/utils/text_utils.py`.
4. **Camada de Adaptadores:** Os serviços concretos em `src/services/` foram refatorados como adaptadores limpos que assinam os contratos do Core de forma implícita (structural typing).

## Consequências

### Positivas
* **Desacoplamento Completo:** A regra de negócios do pipeline de portaria está totalmente isolada de bibliotecas externas e frameworks de deep learning.
* **Testabilidade Unitária Simplificada:** Agora é possível testar exaustivamente o comportamento do caso de uso injetando objetos mock simples que implementam as interfaces (`typing.Protocol`), sem carregar pesos de modelos pesados em memória na execução de testes.
* **Conformidade com DIP e OCP:** Substituir o motor de OCR ou de detecção YOLO agora exige apenas a criação de uma nova classe adaptadora de cerca de 20 linhas, sem qualquer alteração nas regras de negócio ou de orquestração.

### Negativas
* **Aumento no Número de Arquivos:** O projeto possui uma estrutura de diretórios mais ramificada e um número ligeiramente maior de arquivos de classes e scripts auxiliares.
