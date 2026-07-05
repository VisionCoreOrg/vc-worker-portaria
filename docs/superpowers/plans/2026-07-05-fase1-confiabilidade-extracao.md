# Fase 1 — Confiabilidade da Extração de Placas: Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar os itens 1.1–1.4 (+ parte worker do 1.5) do plano de confiabilidade LPR (`../../../relatorios/2026-07-05-confiabilidade-lpr/03-plano-de-evolucao.md` na raiz do workspace): allowlist + seleção de caixas no OCR, janela deslizante com validação de formato BR, status `revisar`, OCR multi-variante com voto e propagação da confiança do OCR — medido contra o baseline offline.

**Architecture:** O domínio (`text_utils`) ganha `extrair_placa` (janela deslizante + mapa posicional + regex BR) e `escolher_leitura` (voto entre leituras candidatas). O adapter OCR passa a rodar o EasyOCR em 3 variantes do crop (cinza+CLAHE, Otsu, Otsu invertida) com allowlist e devolve uma lista de leituras `(texto_cru, confianca)` — a escolha é regra de domínio no caso de uso. O caso de uso decide `sucesso`/`revisar`/`filtrado` e envia `confianca_ocr` no payload.

**Tech Stack:** Python 3.12, EasyOCR (já instalado), OpenCV, pytest. Nenhuma dependência nova.

## Global Constraints

- **Repo git:** `vc-worker-portaria` (o workspace raiz NÃO é repo). Branch: `feat/fase1-confiabilidade-extracao` a partir de `main`.
- **Testes unitários:** `python3 -m pytest tests/ -v` na raiz do repo (python do sistema tem pytest/easyocr/cv2/numpy). **NUNCA importar `src.services.ia_service` em teste unitário** — `onnxruntime` só existe no container.
- **Avaliação end-to-end:** só dentro do container `vc_worker_portaria` (Task 6). Baseline a bater: **acurácia estrita 26,3% / CER 33,5%** (`eval/results/eval_20260705T174011Z.json`, 38 imagens).
- **API Core (fato verificado):** `RegistroCreate` (vc-api-core `src/modules/registros/schemas.py`) tipa `status` como `str` livre e o Pydantic ignora campos extras → enviar `status="revisar"` e `confianca_ocr` é seguro. Persistir `confianca_ocr` no banco e exibir `revisar` no dashboard ficam FORA deste plano (follow-up no vc-api-core/vc-frontend).
- **Idioma:** código, comentários, docstrings e commits em PT-BR, no estilo do repo (conventional commits: `feat:`, `test:`, `chore:`).
- **A suite deve ficar verde ao fim de cada task.** Entre as Tasks 3 e 4 o `main.py` de produção fica temporariamente inconsistente (contrato do OCRReader muda antes do use case) — aceitável em feature branch; não fazer deploy até a Task 6.
- Formato de placa BR (único regex, cobre antigo e Mercosul): `^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$`.

## Estrutura de arquivos (visão geral)

| Arquivo | Ação | Responsabilidade após a Fase 1 |
|---|---|---|
| `src/core/text_utils.py` | Reescrever | Domínio puro: normalização ASCII, mapa posicional, regex BR, `extrair_placa`, `escolher_leitura`, dataclasses `Extracao`/`Decisao` |
| `src/utils/image_utils.py` | Modificar | `letterbox`/`nms` intactos; + `variantes_para_ocr`; `pre_processar_imagem_ocr` removida na Task 3 |
| `src/services/ocr_service.py` | Reescrever | EasyOCR com allowlist sobre 3 variantes; devolve lista de leituras candidatas + binarizada p/ upload |
| `src/core/interfaces.py` | Modificar | Novo contrato do Protocol `OCRReader` |
| `src/core/use_cases.py` | Modificar | Decisão sucesso/revisar/filtrado via `escolher_leitura`; payload com `confianca_ocr` |
| `src/config.py` + `src/main.py` | Modificar | `OCR_CONF_MINIMA_SUCESSO` (env) injetada no use case |
| `eval/run_eval.py` | Modificar | Mesmo caminho de código novo; colunas `conf_ocr`/`status` |
| `tests/test_text_utils.py` | Reescrever | Testes do domínio novo (casos reais do baseline) |
| `tests/test_image_utils.py` | Criar | Testes das variantes |
| `tests/test_ocr_service.py` | Criar | Testes do adapter com leitor fake injetado |
| `tests/test_use_cases.py` | Criar | Testes da decisão de status e payload com fakes dos 4 Protocols |

---

### Task 1: Domínio — `extrair_placa` e `escolher_leitura`

**Files:**
- Modify: `src/core/text_utils.py` (reescrita completa)
- Modify: `tests/test_text_utils.py` (reescrita completa)

**Interfaces:**
- Consumes: nada (domínio puro, stdlib apenas).
- Produces (usado pelas Tasks 4 e 5):
  - `@dataclass Extracao(placa: str, valida: bool, correcoes: int)`
  - `@dataclass Decisao(placa: str, valida: bool, confianca_ocr: float, texto_bruto: str)`
  - `extrair_placa(texto_ocr: str) -> Extracao`
  - `escolher_leitura(leituras: list[tuple[str, float]]) -> Decisao`
  - `normalizar_texto_ocr(texto: str) -> str`, `corrigir_janela(janela: str) -> str`, `eh_formato_valido(placa: str) -> bool`
  - A função antiga `corrigir_placa` **deixa de existir** (era a heurística "últimos 7").

- [ ] **Step 0: Criar branch e commitar o harness existente**

```bash
cd /home/felipe/Projetos/VisionCore/vc-worker-portaria
git checkout -b feat/fase1-confiabilidade-extracao
git add eval/
git commit -m "chore: adiciona harness de avaliação offline com baseline 26,3% (38 imagens)"
```

- [ ] **Step 1: Escrever os testes que devem falhar**

Substituir `tests/test_text_utils.py` inteiro por:

```python
"""Testes das regras de domínio de extração de placas brasileiras."""

from src.core.text_utils import (
    Decisao,
    Extracao,
    corrigir_janela,
    eh_formato_valido,
    escolher_leitura,
    extrair_placa,
    normalizar_texto_ocr,
)

# --- normalizar_texto_ocr ---------------------------------------------------

def test_normaliza_uppercase_e_remove_simbolos():
    assert normalizar_texto_ocr("br abc-1234!") == "BRABC1234"


def test_remove_alfanumericos_unicode():
    # str.isalnum() aceitaria 'ª' — regressão do caso real 'KYXDD28ª' (7.jpg)
    assert normalizar_texto_ocr("KYXDD28ª") == "KYXDD28"


# --- corrigir_janela ---------------------------------------------------------

def test_corrige_digitos_no_prefixo_de_letras():
    assert corrigir_janela("48C1234") == "ABC1234"


def test_corrige_letras_nas_posicoes_de_digito():
    assert corrigir_janela("ABCI2Z4") == "ABC1224"


def test_posicao_4_nao_e_alterada():
    # Pode ser letra (Mercosul) ou dígito (antiga) — forçar quebraria um formato
    assert corrigir_janela("ABC1O23") == "ABC1O23"
    assert corrigir_janela("ABC1023") == "ABC1023"


# --- eh_formato_valido -------------------------------------------------------

def test_valida_formato_antigo_e_mercosul():
    assert eh_formato_valido("ABC1234")
    assert eh_formato_valido("ABC1D23")


def test_rejeita_formatos_impossiveis():
    # Regressão do caso real '7UB5D38' (10.jpg), aceito pelo filtro antigo de len==7
    assert not eh_formato_valido("7UB5D38")
    assert not eh_formato_valido("AB1C234")
    assert not eh_formato_valido("ABC12345")
    assert not eh_formato_valido("")


# --- extrair_placa -----------------------------------------------------------

def test_placa_exata_permanece():
    assert extrair_placa("ABC1234") == Extracao("ABC1234", True, 0)


def test_minusculas_sao_normalizadas():
    assert extrair_placa("abc1d23") == Extracao("ABC1D23", True, 0)


def test_ignora_prefixo_brasil():
    # Regressão: bruto real 'BRasILPOX4G21' (38.jpg) virava 'XAG218R' com os últimos-7
    ext = extrair_placa("BRasILPOX4G21")
    assert ext.placa == "POX4G21"
    assert ext.valida


def test_ignora_prefixo_aplicando_correcao():
    # Caso real (8.jpg): '191[00V0d55' → OOV0D55 (0→O duas vezes no prefixo de letras)
    ext = extrair_placa("191[00V0d55")
    assert ext.placa == "OOV0D55"
    assert ext.valida
    assert ext.correcoes == 2


def test_janela_com_menos_correcoes_vence():
    # Em '5HRFB4D54' (11.jpg), 'HRFB4D5' também validaria com 2 correções;
    # 'RFB4D54' vence com 0
    ext = extrair_placa("5hRFB4D54")
    assert ext.placa == "RFB4D54"
    assert ext.correcoes == 0


def test_empate_escolhe_janela_mais_a_direita():
    # Lixo observado nos dados é majoritariamente prefixo ('BRASIL', sujeira)
    ext = extrair_placa("ABC1234DEF5678")
    assert ext.placa == "DEF5678"


def test_texto_curto_retorna_invalido_sem_correcao():
    assert extrair_placa("AK1JD") == Extracao("AK1JD", False, 0)
    assert extrair_placa("") == Extracao("", False, 0)


# --- escolher_leitura --------------------------------------------------------

def test_lista_vazia_retorna_decisao_vazia():
    assert escolher_leitura([]) == Decisao("", False, 0.0, "")


def test_leitura_valida_vence_invalida_mesmo_com_conf_menor():
    decisao = escolher_leitura([("F172", 0.9), ("BRASILABC1234", 0.6)])
    assert decisao.placa == "ABC1234"
    assert decisao.valida
    assert decisao.confianca_ocr == 0.6
    assert decisao.texto_bruto == "BRASILABC1234"


def test_menos_correcoes_vence_confianca():
    decisao = escolher_leitura([("ABC1234", 0.5), ("48C1234", 0.9)])
    assert decisao.placa == "ABC1234"
    assert decisao.texto_bruto == "ABC1234"


def test_confianca_desempata_leituras_equivalentes():
    decisao = escolher_leitura([("ABC1234", 0.5), ("XYZ0A11", 0.9)])
    assert decisao.placa == "XYZ0A11"


def test_nenhuma_valida_retorna_melhor_esforco():
    decisao = escolher_leitura([("F172", 0.9)])
    assert decisao.placa == "F172"
    assert not decisao.valida
```

- [ ] **Step 2: Rodar e confirmar que falham**

Run: `python3 -m pytest tests/test_text_utils.py -v`
Expected: FAIL na coleta com `ImportError: cannot import name 'Decisao' from 'src.core.text_utils'`

- [ ] **Step 3: Implementar o domínio**

Substituir `src/core/text_utils.py` inteiro por:

```python
"""Regras de domínio para extração e correção de placas brasileiras."""

import re
from dataclasses import dataclass

# Dicionários de mapeamento de caracteres ambíguos no OCR de placas brasileiras.
# Nas posições de dígito, 'Q' e 'D' mapeiam ambos para '0' intencionalmente: são
# as duas confusões de OCR mais comuns com o zero e não há mapeamento melhor.
dict_int_para_letra = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
dict_letra_para_int = {'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'Q': '0', 'D': '0'}

# Padrão único que cobre placa antiga (AAA0000) e Mercosul (AAA0A00):
# a posição 4 aceita letra ou dígito.
PADRAO_PLACA_BR = re.compile(r"^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$")
_NAO_ALFANUM_ASCII = re.compile(r"[^A-Z0-9]")


@dataclass
class Extracao:
    placa: str      # melhor janela corrigida (ou o texto normalizado, se < 7 chars)
    valida: bool    # True se casa com PADRAO_PLACA_BR
    correcoes: int  # quantos caracteres o mapa de ambiguidade alterou


@dataclass
class Decisao:
    placa: str
    valida: bool
    confianca_ocr: float
    texto_bruto: str  # leitura crua que originou a placa escolhida


def normalizar_texto_ocr(texto: str) -> str:
    """Uppercase + remove tudo que não for ASCII A-Z/0-9.

    Não usar str.isalnum(): ele aceita alfanuméricos Unicode (ex.: 'ª'),
    que já contaminaram placas gravadas no banco.
    """
    return _NAO_ALFANUM_ASCII.sub("", texto.upper())


def corrigir_janela(janela: str) -> str:
    """Aplica o mapa posicional de ambiguidade a uma janela de exatamente 7 chars."""
    corrigida = []
    for i, char in enumerate(janela):
        if i in (0, 1, 2):
            corrigida.append(dict_int_para_letra.get(char, char))
        elif i in (3, 5, 6):
            corrigida.append(dict_letra_para_int.get(char, char))
        else:  # posição 4: letra (Mercosul) ou dígito (antiga) — não forçar
            corrigida.append(char)
    return "".join(corrigida)


def eh_formato_valido(placa: str) -> bool:
    return bool(PADRAO_PLACA_BR.fullmatch(placa))


def extrair_placa(texto_ocr: str) -> Extracao:
    """Encontra a melhor janela de 7 caracteres dentro do texto cru do OCR.

    Substitui a heurística antiga de "últimos 7 chars", que desalinhava a
    leitura quando havia texto extra no crop (ex.: 'BRASIL', molduras de
    concessionária). Critério de escolha entre janelas: formato válido >
    menos correções aplicadas > mais à direita (o lixo observado nos dados
    reais é majoritariamente prefixo).
    """
    texto = normalizar_texto_ocr(texto_ocr)
    if len(texto) < 7:
        return Extracao(placa=texto, valida=False, correcoes=0)

    melhor_chave = None
    melhor = None
    for i in range(len(texto) - 6):
        janela = texto[i:i + 7]
        corrigida = corrigir_janela(janela)
        correcoes = sum(1 for a, b in zip(janela, corrigida) if a != b)
        chave = (eh_formato_valido(corrigida), -correcoes, i)
        if melhor_chave is None or chave > melhor_chave:
            melhor_chave = chave
            melhor = Extracao(placa=corrigida, valida=chave[0], correcoes=correcoes)
    return melhor


def escolher_leitura(leituras: list[tuple[str, float]]) -> Decisao:
    """Escolhe a melhor leitura entre candidatas (texto_cru, confianca_ocr).

    Critério: extração em formato válido > menos correções > maior
    confiança do OCR.
    """
    if not leituras:
        return Decisao(placa="", valida=False, confianca_ocr=0.0, texto_bruto="")

    melhor_chave = None
    melhor = None
    for texto, confianca in leituras:
        extracao = extrair_placa(texto)
        chave = (extracao.valida, -extracao.correcoes, confianca)
        if melhor_chave is None or chave > melhor_chave:
            melhor_chave = chave
            melhor = Decisao(
                placa=extracao.placa,
                valida=extracao.valida,
                confianca_ocr=confianca,
                texto_bruto=texto,
            )
    return melhor
```

- [ ] **Step 4: Rodar os testes do módulo**

Run: `python3 -m pytest tests/test_text_utils.py -v`
Expected: 17 PASSED

- [ ] **Step 5: Rodar a suite inteira**

Run: `python3 -m pytest tests/ -v`
Expected: tudo PASSED (os testes antigos de `corrigir_placa` foram substituídos neste task; `use_cases` ainda importa `corrigir_placa` mas nenhum teste o importa — se a coleta falhar por isso, é sinal de import indevido em teste, investigar antes de seguir)

**Atenção:** `src/core/use_cases.py` ainda referencia `corrigir_placa` (removida). Isso quebra `import src.core.use_cases` até a Task 4 — não há teste que o importe antes disso. Não rodar o worker de produção neste intervalo.

- [ ] **Step 6: Commit**

```bash
git add src/core/text_utils.py tests/test_text_utils.py
git commit -m "feat: janela deslizante com validação de formato BR substitui heurística últimos-7"
```

---

### Task 2: Variantes de pré-processamento para OCR

**Files:**
- Modify: `src/utils/image_utils.py` (adicionar função; manter `letterbox`, `nms` e `pre_processar_imagem_ocr` intactas por ora)
- Create: `tests/test_image_utils.py`

**Interfaces:**
- Consumes: nada novo.
- Produces (usado pela Task 3): `variantes_para_ocr(img: np.ndarray) -> list[tuple[str, np.ndarray]]` — lista ordenada `[("cinza_clahe", …), ("otsu", …), ("otsu_invertida", …)]`, todas grayscale uint8 ampliadas 2×.

- [ ] **Step 1: Escrever os testes que devem falhar**

Criar `tests/test_image_utils.py`:

```python
"""Testes das variantes de pré-processamento para OCR multi-variante."""

import numpy as np

from src.utils.image_utils import variantes_para_ocr


def _crop_sintetico():
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(40, 120, 3), dtype=np.uint8)


def test_retorna_tres_variantes_na_ordem_esperada():
    nomes = [nome for nome, _ in variantes_para_ocr(_crop_sintetico())]
    assert nomes == ["cinza_clahe", "otsu", "otsu_invertida"]


def test_todas_ampliadas_2x_grayscale_uint8():
    for nome, img in variantes_para_ocr(_crop_sintetico()):
        assert img.shape == (80, 240), nome
        assert img.dtype == np.uint8, nome


def test_otsu_e_binaria_e_invertida_e_complemento():
    variantes = dict(variantes_para_ocr(_crop_sintetico()))
    assert set(np.unique(variantes["otsu"])) <= {0, 255}
    assert (variantes["otsu_invertida"] == 255 - variantes["otsu"]).all()
```

- [ ] **Step 2: Rodar e confirmar que falham**

Run: `python3 -m pytest tests/test_image_utils.py -v`
Expected: FAIL na coleta com `ImportError: cannot import name 'variantes_para_ocr'`

- [ ] **Step 3: Implementar**

Adicionar ao final de `src/utils/image_utils.py`:

```python
def variantes_para_ocr(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Gera variantes de pré-processamento do crop para OCR multi-variante.

    A binarização de Otsu global destrói placas legíveis sob iluminação
    irregular e escolhe a polaridade sozinha conforme o histograma; por isso
    o OCR roda também na variante em cinza (sem threshold) e na binarizada
    invertida — a melhor leitura é escolhida no domínio (escolher_leitura).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    largura = int(gray_clahe.shape[1] * 2)
    altura = int(gray_clahe.shape[0] * 2)
    ampliada = cv2.resize(gray_clahe, (largura, altura), interpolation=cv2.INTER_CUBIC)

    suave = cv2.bilateralFilter(ampliada, d=5, sigmaColor=75, sigmaSpace=75)
    _, otsu = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [
        ("cinza_clahe", suave),
        ("otsu", otsu),
        ("otsu_invertida", cv2.bitwise_not(otsu)),
    ]
```

- [ ] **Step 4: Rodar os testes**

Run: `python3 -m pytest tests/test_image_utils.py tests/test_text_utils.py -v`
Expected: tudo PASSED

- [ ] **Step 5: Commit**

```bash
git add src/utils/image_utils.py tests/test_image_utils.py
git commit -m "feat: variantes de pré-processamento (cinza+CLAHE, Otsu, Otsu invertida) para OCR"
```

---

### Task 3: Adapter OCR multi-variante com allowlist

**Files:**
- Modify: `src/services/ocr_service.py` (reescrita completa)
- Modify: `src/core/interfaces.py:14-23` (Protocol `OCRReader`)
- Modify: `src/utils/image_utils.py` (remover `pre_processar_imagem_ocr`, agora sem consumidor)
- Create: `tests/test_ocr_service.py`

**Interfaces:**
- Consumes: `variantes_para_ocr` (Task 2).
- Produces (usado pelas Tasks 4 e 5): `EasyOCRReader.ler_texto(crop: np.ndarray) -> tuple[list[tuple[str, float]], np.ndarray]` — lista de leituras candidatas `(texto_cru, confianca)` + imagem binarizada (variante "otsu") para upload de auditoria. Constantes `ALLOWLIST_PLACA` e `MIN_CHARS_CANDIDATA`.

- [ ] **Step 1: Escrever os testes que devem falhar**

Criar `tests/test_ocr_service.py`:

```python
"""Testes do adapter EasyOCR multi-variante com leitor fake injetado."""

import numpy as np

from src.services.ocr_service import ALLOWLIST_PLACA, EasyOCRReader


def _crop():
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(40, 120, 3), dtype=np.uint8)


def _caixa(x, texto, conf):
    # bbox no formato do EasyOCR: 4 pontos [x, y]
    return ([[x, 0], [x + 50, 0], [x + 50, 10], [x, 10]], texto, conf)


class FakeLeitor:
    """Devolve um resultado pré-definido por chamada (uma por variante)."""

    def __init__(self, resultados):
        self._resultados = list(resultados)
        self.allowlists = []

    def readtext(self, img, allowlist=None):
        self.allowlists.append(allowlist)
        return self._resultados.pop(0)


def test_passa_allowlist_em_todas_as_variantes():
    fake = FakeLeitor([[], [], []])
    EasyOCRReader(fake).ler_texto(_crop())
    assert fake.allowlists == [ALLOWLIST_PLACA] * 3


def test_concatena_caixas_em_ordem_esquerda_direita():
    # Regressão do caso real 24.jpg: caixas fora de ordem viravam 'J17...QEX7'
    fake = FakeLeitor([[_caixa(100, "J17", 0.8), _caixa(0, "QEX7", 0.9)], [], []])
    leituras, _ = EasyOCRReader(fake).ler_texto(_crop())
    assert ("QEX7J17", 0.8) in leituras  # confiança = mínima entre as caixas


def test_caixas_grandes_viram_candidatas_individuais():
    fake = FakeLeitor([[_caixa(0, "BRASIL", 0.5), _caixa(60, "ABC1234", 0.9)], [], []])
    leituras, _ = EasyOCRReader(fake).ler_texto(_crop())
    assert ("ABC1234", 0.9) in leituras
    assert ("BRASIL", 0.5) in leituras
    assert ("BRASILABC1234", 0.5) in leituras


def test_caixa_unica_nao_duplica_candidata():
    fake = FakeLeitor([[_caixa(0, "ABC1234", 0.9)], [], []])
    leituras, _ = EasyOCRReader(fake).ler_texto(_crop())
    assert leituras.count(("ABC1234", 0.9)) == 1


def test_erro_em_uma_variante_nao_derruba_as_demais():
    class LeitorFalhaPrimeira(FakeLeitor):
        def readtext(self, img, allowlist=None):
            if not self.allowlists:
                self.allowlists.append(allowlist)
                raise RuntimeError("boom")
            return super().readtext(img, allowlist=allowlist)

    fake = LeitorFalhaPrimeira([[_caixa(0, "ABC1234", 0.9)], []])
    leituras, _ = EasyOCRReader(fake).ler_texto(_crop())
    assert ("ABC1234", 0.9) in leituras


def test_retorna_imagem_binarizada_para_upload():
    fake = FakeLeitor([[], [], []])
    _, binarizada = EasyOCRReader(fake).ler_texto(_crop())
    assert set(np.unique(binarizada)) <= {0, 255}
```

- [ ] **Step 2: Rodar e confirmar que falham**

Run: `python3 -m pytest tests/test_ocr_service.py -v`
Expected: FAIL na coleta com `ImportError: cannot import name 'ALLOWLIST_PLACA'`

- [ ] **Step 3: Reescrever o adapter**

Substituir `src/services/ocr_service.py` inteiro por:

```python
from typing import Tuple
import numpy as np
import easyocr

from src.core.logger import configurar_logger
from src.utils.image_utils import variantes_para_ocr

logger = configurar_logger("EasyOCRReader")

# Placas BR só contêm A-Z e 0-9; restringir o vocabulário do EasyOCR elimina
# pontuação, minúsculas e Unicode (ex.: 'ª') na origem.
ALLOWLIST_PLACA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Caixas com menos caracteres que isso são quase sempre ruído (parafusos,
# moldura); participam da leitura concatenada mas não viram candidata própria.
MIN_CHARS_CANDIDATA = 5


class EasyOCRReader:
    """
    Implementa o leitor OCR usando a biblioteca EasyOCR sobre múltiplas
    variantes de pré-processamento. Conforme com o Protocol 'OCRReader'.
    """

    def __init__(self, leitor: easyocr.Reader):
        self.leitor = leitor

    def ler_texto(self, crop: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Executa OCR sobre as variantes do recorte e devolve leituras candidatas.
        Retorna:
            (leituras, imagem_binarizada) — leituras é uma lista de tuplas
            (texto_cru, confianca_ocr); a escolha da melhor leitura é regra
            de domínio (escolher_leitura) e vive no caso de uso.
        """
        variantes = variantes_para_ocr(crop)
        img_binarizada = dict(variantes)["otsu"]

        leituras: list[tuple[str, float]] = []
        for nome, img in variantes:
            try:
                resultado = self.leitor.readtext(img, allowlist=ALLOWLIST_PLACA)
            except Exception as e:
                logger.error(f"Falha no EasyOCR na variante '{nome}': {e}")
                continue
            if not resultado:
                continue

            # O EasyOCR não garante ordem de leitura entre caixas — ordenar
            # da esquerda para a direita antes de concatenar.
            caixas = sorted(resultado, key=lambda r: min(p[0] for p in r[0]))
            concatenado = "".join(texto for _, texto, _ in caixas)
            conf_minima = min(float(conf) for _, _, conf in caixas)
            leituras.append((concatenado, conf_minima))

            if len(caixas) > 1:
                for _, texto, conf in caixas:
                    if len(texto) >= MIN_CHARS_CANDIDATA:
                        leituras.append((texto, float(conf)))

        return leituras, img_binarizada
```

- [ ] **Step 4: Atualizar o Protocol**

Em `src/core/interfaces.py`, substituir a classe `OCRReader` (linhas 14–23) por:

```python
class OCRReader(Protocol):
    def ler_texto(self, crop: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Executa OCR sobre variantes de pré-processamento do recorte da placa.
        Retorna:
            (leituras, imagem_binarizada) — leituras é uma lista de tuplas
            (texto_cru, confianca_ocr) candidatas, uma por variante e por
            caixa de texto relevante. A escolha da melhor leitura é regra de
            domínio (escolher_leitura) e vive no caso de uso.
        """
        ...
```

- [ ] **Step 5: Remover `pre_processar_imagem_ocr`**

Em `src/utils/image_utils.py`, apagar a função `pre_processar_imagem_ocr` inteira (ficou sem consumidor: o adapter usa `variantes_para_ocr`).

- [ ] **Step 6: Rodar a suite inteira**

Run: `python3 -m pytest tests/ -v`
Expected: tudo PASSED

- [ ] **Step 7: Commit**

```bash
git add src/services/ocr_service.py src/core/interfaces.py src/utils/image_utils.py tests/test_ocr_service.py
git commit -m "feat: OCR multi-variante com allowlist e leituras candidatas com confiança"
```

---

### Task 4: Caso de uso — decisão sucesso/revisar/filtrado e payload

**Files:**
- Modify: `src/core/use_cases.py`
- Create: `tests/test_use_cases.py`

**Interfaces:**
- Consumes: `escolher_leitura`/`Decisao` (Task 1); novo contrato `ler_texto` (Task 3).
- Produces (usado pela Task 5): `ProcessarEventoUseCase.__init__(detector, ocr_reader, storage, api_client, camera_id_default="camera_default", conf_minima_sucesso=0.5)`. Payload ganha `"confianca_ocr": float`; `status` pode ser `"sucesso" | "revisar" | "filtrado"`.

- [ ] **Step 1: Escrever os testes que devem falhar**

Criar `tests/test_use_cases.py`:

```python
"""Testes da decisão de status e montagem do payload com fakes dos Protocols."""

import numpy as np

from src.core.use_cases import ProcessarEventoUseCase


class FakeDetector:
    def detectar(self, imagem):
        return np.zeros((10, 30, 3), dtype=np.uint8), 0.9


class FakeOCR:
    def __init__(self, leituras):
        self._leituras = leituras

    def ler_texto(self, crop):
        return self._leituras, np.zeros((10, 30), dtype=np.uint8)


class FakeStorage:
    def baixar_imagem(self, chave):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def upload_recorte(self, imagem, placa, sufixo=""):
        return f"http://minio/{placa}{sufixo}.jpg"


class FakeAPI:
    def __init__(self):
        self.payloads = []

    def registrar_passagem(self, payload):
        self.payloads.append(payload)
        return True


def _executar(leituras, conf_minima=0.5):
    api = FakeAPI()
    caso = ProcessarEventoUseCase(
        detector=FakeDetector(),
        ocr_reader=FakeOCR(leituras),
        storage=FakeStorage(),
        api_client=api,
        conf_minima_sucesso=conf_minima,
    )
    caso.executar({"path": "dataset/1.jpg", "camera_id": "cam_teste"})
    assert len(api.payloads) == 1
    return api.payloads[0]


def test_leitura_valida_confiante_vira_sucesso():
    payload = _executar([("BRASILABC1234", 0.9)])
    assert payload["status"] == "sucesso"
    assert payload["placa"] == "ABC1234"
    assert payload["confianca_ocr"] == 0.9
    assert payload["motivo_filtro"] is None


def test_leitura_valida_com_conf_baixa_vira_revisar():
    payload = _executar([("ABC1234", 0.3)])
    assert payload["status"] == "revisar"
    assert payload["placa"] == "ABC1234"
    assert "0.30" in payload["motivo_filtro"]


def test_nenhuma_leitura_valida_vira_filtrado():
    payload = _executar([("F172", 0.9)])
    assert payload["status"] == "filtrado"
    assert payload["placa"] == "F172"
    assert "melhor esforco" in payload["motivo_filtro"]


def test_ocr_sem_leituras_vira_filtrado_com_motivo_especifico():
    payload = _executar([])
    assert payload["status"] == "filtrado"
    assert payload["placa"] == "—"
    assert payload["motivo_filtro"] == "OCR nao identificou nenhum caractere"


def test_payload_mantem_confianca_yolo_separada_da_ocr():
    payload = _executar([("ABC1234", 0.7)])
    assert payload["confianca"] == 0.9      # YOLO (detector)
    assert payload["confianca_ocr"] == 0.7  # OCR (leitura)
```

- [ ] **Step 2: Rodar e confirmar que falham**

Run: `python3 -m pytest tests/test_use_cases.py -v`
Expected: FAIL na coleta com `ImportError: cannot import name 'corrigir_placa'` (o `use_cases.py` atual importa a função removida na Task 1)

- [ ] **Step 3: Atualizar o caso de uso**

Em `src/core/use_cases.py`:

(a) trocar o import de domínio:

```python
from src.core.text_utils import escolher_leitura
```

(b) substituir o `__init__` por:

```python
    def __init__(
        self,
        detector: Detector,
        ocr_reader: OCRReader,
        storage: StorageRepository,
        api_client: APIClient,
        camera_id_default: str = "camera_default",
        conf_minima_sucesso: float = 0.5,
    ):
        self.detector = detector
        self.ocr_reader = ocr_reader
        self.storage = storage
        self.api_client = api_client
        self.camera_id_default = camera_id_default
        self.conf_minima_sucesso = conf_minima_sucesso
```

(c) substituir os passos 3 e 4 do `executar` (do comentário `# 3. Inferência de OCR` até a linha do `logger.info(f"Filtro Aplicado ...")` inclusive) por:

```python
        # 3. OCR multi-variante (adapter devolve leituras cruas candidatas)
        leituras, img_binarizada = self.ocr_reader.ler_texto(placa_crop)

        # 4. Regras de Domínio: melhor leitura + validação do formato BR
        decisao = escolher_leitura(leituras)

        if decisao.valida and decisao.confianca_ocr >= self.conf_minima_sucesso:
            status = "sucesso"
            motivo_filtro = None
        elif decisao.valida:
            status = "revisar"
            motivo_filtro = f"Confianca OCR baixa ({decisao.confianca_ocr:.2f})"
        elif decisao.placa:
            status = "filtrado"
            motivo_filtro = f"Nenhuma leitura em formato BR (melhor esforco: '{decisao.placa[:20]}')"
        else:
            status = "filtrado"
            motivo_filtro = "OCR nao identificou nenhum caractere"

        placa_salvar = decisao.placa if decisao.placa else "—"

        logger.info(
            f"Filtro Aplicado -> [{status.upper()}] Placa final: {placa_salvar} "
            f"(YOLO: {confianca_yolo:.2f} | OCR: {decisao.confianca_ocr:.2f})"
        )
```

(d) no dicionário `payload`, adicionar logo após a linha `"confianca": ...`:

```python
            "confianca_ocr": round(float(decisao.confianca_ocr), 4),
```

- [ ] **Step 4: Rodar a suite inteira**

Run: `python3 -m pytest tests/ -v`
Expected: tudo PASSED

- [ ] **Step 5: Commit**

```bash
git add src/core/use_cases.py tests/test_use_cases.py
git commit -m "feat: decisão sucesso/revisar/filtrado por formato BR e confiança do OCR no payload"
```

---

### Task 5: Wiring (config + main) e harness de avaliação

**Files:**
- Modify: `src/config.py`
- Modify: `src/main.py:62-68`
- Modify: `eval/run_eval.py`

**Interfaces:**
- Consumes: `conf_minima_sucesso` (Task 4); `escolher_leitura`/`Decisao` (Task 1); novo `ler_texto` (Task 3).
- Produces: env var `OCR_CONF_MINIMA_SUCESSO` (default `0.5`); relatório do eval com colunas `conf_ocr` e status `revisar`.

- [ ] **Step 1: Adicionar a configuração**

Ao final de `src/config.py`:

```python
# ===========================================================================
# OCR — Fase 1 do plano de confiabilidade
# Leituras em formato BR válido com confiança abaixo deste limiar recebem
# status "revisar" (fila de anotação humana) em vez de "sucesso".
# ===========================================================================
OCR_CONF_MINIMA_SUCESSO = float(os.getenv("OCR_CONF_MINIMA_SUCESSO", "0.5"))
```

- [ ] **Step 2: Injetar no main**

Em `src/main.py`, adicionar `OCR_CONF_MINIMA_SUCESSO` ao import de `src.config` e substituir a construção do use case por:

```python
    use_case = ProcessarEventoUseCase(
        detector=detector,
        ocr_reader=ocr_reader,
        storage=storage,
        api_client=api_client,
        camera_id_default=CAMERA_ID,
        conf_minima_sucesso=OCR_CONF_MINIMA_SUCESSO,
    )
```

- [ ] **Step 3: Atualizar o harness**

Em `eval/run_eval.py`:

(a) trocar o import de domínio:

```python
from src.core.text_utils import escolher_leitura  # noqa: E402
```

(b) em `avaliar()`, substituir o trecho do OCR (das linhas `texto_bruto, _ = ocr_reader.ler_texto(crop)` até o `resultados.append({...})` final) por:

```python
        leituras, _ = ocr_reader.ler_texto(crop)
        decisao = escolher_leitura(leituras)
        if decisao.valida and decisao.confianca_ocr >= conf_minima_sucesso:
            status = "sucesso"
        elif decisao.valida:
            status = "revisar"
        else:
            status = "filtrado"
        resultados.append({
            "arquivo": arquivo,
            "gt": placa_real,
            "bruto": decisao.texto_bruto,
            "lido": decisao.placa,
            "conf_yolo": round(float(conf_yolo), 3),
            "conf_ocr": round(float(decisao.confianca_ocr), 3),
            "status": status,
            "dist": levenshtein(decisao.placa, placa_real),
        })
```

(c) mudar a assinatura para `def avaliar(detector, ocr_reader, gt, images_dir, conf_minima_sucesso=0.5):` e, no `main()`, chamar com `conf_minima_sucesso=OCR_CONF_MINIMA_SUCESSO` (importar junto de `USE_GPU`: `from src.config import USE_GPU, OCR_CONF_MINIMA_SUCESSO`).

(d) na tabela de `imprimir_relatorio`, adicionar a coluna `conf_ocr` ao cabeçalho e à linha:

```python
    print(f"\n{'arquivo':<{largura}}  {'gt':<8} {'lido':<8} {'bruto':<14} {'yolo':<6} {'ocr':<6} {'dist':<4} status")
```

e na linha por resultado:

```python
        print(f"{r['arquivo']:<{largura}}  {r['gt']:<8} {str(r['lido']):<8} "
              f"{str(r.get('bruto', ''))[:14]:<14} {r['conf_yolo']:<6} {r.get('conf_ocr', '—'):<6} "
              f"{r['dist']:<4} {marca} {r['status']}")
```

(e) no `resumo`, contar `revisar` junto dos sucessos de formato: adicionar a chave `"n_revisar": sum(1 for r in validos if r["status"] == "revisar")` e incluir `r["status"] in ("sucesso", "revisar")` onde hoje filtra `== "sucesso"` para `sucessos` (a acurácia estrita geral não muda de definição). Imprimir a linha `print(f"Revisar: {resumo['n_revisar']}")` junto do bloco de filtrados. **Atenção:** o caso `sem_deteccao` (crop None) continua igual — não passa pelo OCR.

- [ ] **Step 4: Verificação local (sintaxe apenas — onnxruntime não existe fora do container)**

Run: `python3 -m py_compile eval/run_eval.py src/main.py src/config.py && python3 -m pytest tests/ -v`
Expected: py_compile silencioso; suite toda PASSED

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/main.py eval/run_eval.py
git commit -m "feat: limiar de revisão configurável e harness alinhado ao pipeline multi-variante"
```

---

### Task 6: Validação integrada no container e registro dos resultados

**Files:**
- Modify: `CLAUDE.md` (seções do pipeline e do OCR, desatualizadas após a Fase 1)
- Create: `eval/results/eval_<timestamp>.json` (gerado pelo run)
- Modify (fora do repo, no workspace raiz): `../relatorios/2026-07-05-confiabilidade-lpr/README.md`

**Interfaces:**
- Consumes: todo o pipeline novo (Tasks 1–5).
- Produces: número pós-Fase 1 vs baseline 26,3%/33,5%.

- [ ] **Step 1: Rodar a suite completa uma última vez**

Run: `python3 -m pytest tests/ -v`
Expected: tudo PASSED (≈35 testes)

- [ ] **Step 2: Subir os containers necessários**

```bash
docker start parking_redis && sleep 3 && docker start vc_worker_portaria && sleep 5
docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'redis|worker'
```
Expected: ambos `Up`. (O worker morre sozinho se o Redis não estiver de pé — subir o Redis primeiro.)

- [ ] **Step 3: Copiar código novo, harness e dataset**

```bash
# garantir que os destinos existem (container recriado não os tem)
docker exec vc_worker_portaria mkdir -p /app/eval /app/dataset
# usar 'dir/.' para não aninhar (docker cp de dir para dir existente cria subdir)
docker cp src/. vc_worker_portaria:/app/src/
docker cp eval/. vc_worker_portaria:/app/eval/
docker cp ../dataset/. vc_worker_portaria:/app/dataset/
```

- [ ] **Step 4: Rodar a avaliação**

Run: `docker exec -w /app vc_worker_portaria python eval/run_eval.py`
Expected: tabela com 38 imagens + resumo. **Gate de aceite: acurácia estrita > 26,3% e CER < 33,5%** (expectativa do plano: 55–70% estrita). O tempo sobe ~3× (3 variantes de OCR por imagem) — normal.

Se o gate falhar: NÃO commitar resultados; usar a skill superpowers:systematic-debugging sobre os casos que regrediram (a tabela por imagem mostra exatamente quais) antes de prosseguir.

- [ ] **Step 5: Trazer o resultado e derrubar os containers**

```bash
docker cp "vc_worker_portaria:/app/eval/results/." eval/results/
docker stop vc_worker_portaria parking_redis
```

- [ ] **Step 6: Atualizar a documentação**

(a) Em `CLAUDE.md` do repo, na seção `### ocr_service.py — OCR com Heurísticas BR`, substituir a descrição do pré-processamento fixo e do `corrigir_placa` (últimos-7/len==7) pelo comportamento novo: 3 variantes (`cinza_clahe`, `otsu`, `otsu_invertida`) via `variantes_para_ocr`, allowlist A-Z0-9, leituras candidatas com confiança, `extrair_placa` (janela deslizante + regex `^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$`) e `escolher_leitura` no caso de uso, status `sucesso`/`revisar`/`filtrado` com limiar `OCR_CONF_MINIMA_SUCESSO` (default 0.5). Adicionar `OCR_CONF_MINIMA_SUCESSO` à tabela de variáveis de ambiente.

(b) No workspace raiz, em `relatorios/2026-07-05-confiabilidade-lpr/README.md`, adicionar seção `## Resultados Fase 1` com: acurácia estrita e CER novos vs baseline (26,3% / 33,5%), distribuição 0/1/2/3+, contagem sucesso/revisar/filtrado e o nome do JSON do run.

- [ ] **Step 7: Commit final**

```bash
git add CLAUDE.md eval/results/
git commit -m "feat: resultados da Fase 1 vs baseline e documentação atualizada"
```

- [ ] **Step 8: Encerrar a branch**

Usar a skill superpowers:finishing-a-development-branch para decidir merge em `main` (o repo não usa PR — merges locais, como nas Rodadas anteriores).

---

## Fora do escopo (follow-ups registrados)

- Persistir `confianca_ocr` no banco (coluna + migração Alembic no vc-api-core) e exibir no dashboard; hoje o campo extra é ignorado pelo Pydantic sem erro.
- Frontend: exibir/filtrar o status `revisar` e corrigir o rótulo "Confiança do OCR" (que mostra a confiança do YOLO).
- Deduplicação de reprocessamentos at-least-once nas métricas da API.
- Fase 2 (fusão multi-frame, gate de qualidade) e Fase 3 (OCR especializado) — ver relatório 03.
