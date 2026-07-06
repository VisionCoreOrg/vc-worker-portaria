# Fase 1.5 — Confiança e Consenso Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduzir leituras corretas mandadas para `revisar` (Fix 1) e placas erradas publicadas como `sucesso` (Fix 2) no worker de LPR, sem regredir acurácia — mudanças de princípio, medidas pelo harness sobre 43 imagens.

**Architecture:** Três mudanças pequenas e isoladas. (1) Instrumentação: o harness passa a reportar "sucessos errados". (2) `ocr_service.EasyOCRReader.ler_texto` passa a atribuir à leitura concatenada a confiança da **maior caixa** (a que provavelmente é o corpo da placa) em vez do mínimo global entre todas as caixas — o mínimo era deprimido por caixas de ruído ("BRASIL", moldura). (3) `text_utils.escolher_leitura` passa de argmax para voto: uma placa válida que aparece em ≥2 candidatas vence, via chave `(valida, n_ocorrencias, -correcoes, confianca)`.

**Tech Stack:** Python 3.12, pytest, EasyOCR (fake injetado nos testes), OpenCV.

## Global Constraints

- **Não tocar** em `models/` no container (ONNX + pesos EasyOCR já provisionados).
- Testes rodam com **python do sistema** a partir da raiz do repo (`python3 -m pytest tests/`); **não** importam `src/services/ia_service` (onnxruntime só existe no container). `cv2` está disponível no sistema (4.13.0).
- Mudanças de princípio, **não** caça de limiar sobre as 43 imagens. Não perseguir `29.jpg`/`37.jpg` com hacks — são evidência da Fase 2.
- Regra de status vive só em `text_utils.decidir_status` (compartilhada worker+harness) — não duplicar.
- Baseline oficial (43 imagens, `eval/results/eval_20260706T113448Z.json`): estrita **37.2%** (16/43), CER **29.6%**, `revisar` **15**, **sucessos errados 7**, `sem_deteccao` 5, `filtrado` 2.
- **Gate final (sobre 43, contra o baseline):** estrita ≥ 37.2% **E** CER ≤ 29.6% **E** `revisar` < 15 **E** sucessos errados ≤ 7.

---

### Task 1: Instrumentar "sucessos errados" no harness

O gate precisa deste número e o resumo atual não o traz. "Sucessos errados" = entradas com `status=="sucesso"` e `lido != gt` (só `sucesso`, não `revisar`). Mudança pura em Python, testável sem container.

**Files:**
- Modify: `eval/run_eval.py:83-124` (`imprimir_relatorio`)
- Test: `tests/test_run_eval.py` (criar)

**Interfaces:**
- Consumes: `imprimir_relatorio(resultados: list[dict]) -> dict` (assinatura inalterada).
- Produces: o dict `resumo` retornado passa a incluir a chave `"n_sucessos_errados": int`.

- [ ] **Step 1: Write the failing test**

Criar `tests/test_run_eval.py`:

```python
"""Testes do relatório do harness de avaliação (imprimir_relatorio)."""

from eval.run_eval import imprimir_relatorio


def _rec(arquivo, gt, lido, status, dist):
    return {"arquivo": arquivo, "gt": gt, "lido": lido, "status": status,
            "conf_yolo": 0.9, "conf_ocr": 0.9, "dist": dist, "bruto": lido}


def test_conta_sucessos_errados():
    resultados = [
        _rec("a.jpg", "ABC1234", "ABC1234", "sucesso", 0),  # sucesso correto
        _rec("b.jpg", "ABC1234", "ABX1234", "sucesso", 1),  # sucesso ERRADO
        _rec("c.jpg", "ABC1234", "ABX1234", "revisar", 1),  # errado, mas revisar → não conta
    ]
    resumo = imprimir_relatorio(resultados)
    assert resumo["n_sucessos_errados"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_run_eval.py::test_conta_sucessos_errados -v`
Expected: FAIL com `KeyError: 'n_sucessos_errados'`.

- [ ] **Step 3: Write minimal implementation**

Em `eval/run_eval.py`, dentro de `imprimir_relatorio`, após a linha `acertos_sucesso = [r for r in sucessos if r["lido"] == r["gt"]]` (linha 88), adicionar:

```python
    sucessos_errados = [r for r in validos if r["status"] == "sucesso" and r["lido"] != r["gt"]]
```

No dict `resumo`, adicionar a chave (após `"n_filtrados"`):

```python
        "n_sucessos_errados": len(sucessos_errados),
```

Na impressão do resumo, após a linha `print(f"Revisar: {resumo['n_revisar']}")` (linha 121), adicionar:

```python
    print(f"Sucessos errados (status=sucesso, lido!=gt): {resumo['n_sucessos_errados']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_run_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Run full suite (regressão)**

Run: `python3 -m pytest tests/ -q`
Expected: 48 passed.

- [ ] **Step 6: Commit**

```bash
git add eval/run_eval.py tests/test_run_eval.py
git commit -m "feat(eval): reporta contagem de sucessos errados no harness"
```

---

### Task 2: Fix 1 — confiança da leitura pela maior caixa

Hoje a leitura concatenada recebe `min(conf de todas as caixas)`, incluindo ruído — deprime leituras corretas para `revisar`. Passa a receber a confiança da **maior caixa por nº de caracteres** (proxy do corpo da placa; empate → primeira). Escalonamento documentado abaixo, só se o gate não fechar.

**Files:**
- Modify: `src/services/ocr_service.py:49-54`
- Test: `tests/test_ocr_service.py:36-48` (atualizar 2 testes existentes)

**Interfaces:**
- Consumes: `variantes_para_ocr`, `ALLOWLIST_PLACA`, `MIN_CHARS_CANDIDATA` (inalterados).
- Produces: `EasyOCRReader.ler_texto(crop) -> (list[tuple[str, float]], np.ndarray)` — assinatura inalterada; muda apenas a confiança atribuída à tupla da concatenação: agora `conf da caixa com mais caracteres` em vez do mínimo global.

- [ ] **Step 1: Atualizar os testes existentes para a nova semântica (failing)**

Em `tests/test_ocr_service.py`, substituir `test_concatena_caixas_em_ordem_esquerda_direita` (linhas 36-40) por:

```python
def test_concatena_caixas_em_ordem_esquerda_direita():
    # Regressão do caso real 24.jpg: caixas fora de ordem viravam 'J17...QEX7'
    fake = FakeLeitor([[_caixa(100, "J17", 0.8), _caixa(0, "QEX7", 0.9)], [], []])
    leituras, _ = EasyOCRReader(fake).ler_texto(_crop())
    # confiança = a da MAIOR caixa (QEX7, 4 chars, conf 0.9), não o mínimo global
    assert ("QEX7J17", 0.9) in leituras
```

E `test_caixas_grandes_viram_candidatas_individuais` (linhas 43-48) por:

```python
def test_caixas_grandes_viram_candidatas_individuais():
    fake = FakeLeitor([[_caixa(0, "BRASIL", 0.5), _caixa(60, "ABC1234", 0.9)], [], []])
    leituras, _ = EasyOCRReader(fake).ler_texto(_crop())
    assert ("ABC1234", 0.9) in leituras
    assert ("BRASIL", 0.5) in leituras
    # concatenação herda a conf da MAIOR caixa (ABC1234, 7 chars, 0.9), não 0.5
    assert ("BRASILABC1234", 0.9) in leituras
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ocr_service.py -v`
Expected: FAIL nos 2 testes acima (esperam 0.9, código ainda produz 0.8 / 0.5).

- [ ] **Step 3: Write minimal implementation**

Em `src/services/ocr_service.py`, substituir a linha 53:

```python
            conf_minima = min(float(conf) for _, _, conf in caixas)
            leituras.append((concatenado, conf_minima))
```

por:

```python
            # A confiança da leitura concatenada deve refletir o corpo da placa,
            # não o mínimo global (caixas de ruído como "BRASIL"/moldura deprimem
            # leituras corretas para 'revisar'). Usamos a conf da MAIOR caixa por
            # nº de caracteres — a que provavelmente é a placa; empate → primeira.
            caixa_maior = max(caixas, key=lambda r: len(r[1]))
            conf_leitura = float(caixa_maior[2])
            leituras.append((concatenado, conf_leitura))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ocr_service.py -v`
Expected: PASS (6 testes).

- [ ] **Step 5: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: 48 passed.

- [ ] **Step 6: Commit**

```bash
git add src/services/ocr_service.py tests/test_ocr_service.py
git commit -m "fix(ocr): confianca da leitura pela maior caixa, nao pelo minimo global"
```

**Escalonamento (só se o gate não fechar `revisar < 15`):** baixar `MIN_CHARS_CANDIDATA` de 5 para 3 (placas partidas "ABC"+"1234" viram candidatas próprias). Avaliar no gate, não preventivamente.

---

### Task 3: Fix 2 — consenso entre variantes em escolher_leitura

`escolher_leitura` é argmax por `(valida, -correcoes, confianca)` — "menos correções" às vezes publica placa errada como `sucesso`. Passa a votar: uma placa **válida** que aparece em ≥2 candidatas (contando extrações idênticas) vence via chave `(valida, n_ocorrencias, -correcoes, confianca)`.

**Files:**
- Modify: `src/core/text_utils.py:85-107` (`escolher_leitura`)
- Test: `tests/test_text_utils.py` (adicionar 2 testes na seção `escolher_leitura`, após a linha 128)

**Interfaces:**
- Consumes: `extrair_placa(texto) -> Extracao` (campos `.placa`, `.valida`, `.correcoes`); `Decisao`.
- Produces: `escolher_leitura(leituras: list[tuple[str, float]]) -> Decisao` — assinatura inalterada; muda apenas o critério de desempate (consenso antes de correções/confiança).

- [ ] **Step 1: Write the failing tests**

Em `tests/test_text_utils.py`, adicionar após `test_nenhuma_valida_retorna_melhor_esforco` (linha 128):

```python
def test_consenso_vence_menos_correcoes():
    # Placa correta lida em 2 variantes (2 candidatas idênticas) vence a placa
    # de 1 candidata mesmo que esta tenha menos correções e maior confiança.
    decisao = escolher_leitura([
        ("OOV0D55", 0.4),   # correta, aparece 2x → consenso 2
        ("OOV0D55", 0.4),
        ("VOD5519", 0.9),   # 1 candidata, alta confiança
    ])
    assert decisao.placa == "OOV0D55"


def test_consenso_nao_promove_invalida_sobre_valida():
    # Uma placa INVÁLIDA repetida não vence uma válida única (valida domina).
    decisao = escolher_leitura([
        ("ZZ", 0.9), ("ZZ", 0.9),   # inválidas, 2x
        ("ABC1234", 0.3),           # válida, 1x
    ])
    assert decisao.placa == "ABC1234"
    assert decisao.valida
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_text_utils.py -k consenso -v`
Expected: `test_consenso_vence_menos_correcoes` FAIL (código atual escolhe `VOD5519` por menos correções/maior conf). `test_consenso_nao_promove_invalida` já passa (valida domina) — aceitável.

- [ ] **Step 3: Write implementation**

Em `src/core/text_utils.py`, substituir o corpo de `escolher_leitura` (linhas 91-107) por:

```python
    if not leituras:
        return Decisao(placa="", valida=False, confianca_ocr=0.0, texto_bruto="")

    # Pré-computa a extração de cada candidata e conta ocorrências da MESMA
    # placa válida entre todas as candidatas — o consenso na prática: a mesma
    # placa lida em 2 variantes gera 2 candidatas que extraem igual.
    extraidas = [extrair_placa(texto) for texto, _ in leituras]
    ocorrencias: dict[str, int] = {}
    for ext in extraidas:
        if ext.valida:
            ocorrencias[ext.placa] = ocorrencias.get(ext.placa, 0) + 1

    melhor_chave = None
    melhor = None
    for (texto, confianca), extracao in zip(leituras, extraidas):
        n_ocorrencias = ocorrencias.get(extracao.placa, 0) if extracao.valida else 0
        chave = (extracao.valida, n_ocorrencias, -extracao.correcoes, confianca)
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

Atualizar a docstring de `escolher_leitura` (linhas 86-90) para:

```python
    """Escolhe a melhor leitura entre candidatas (texto_cru, confianca_ocr).

    Critério: extração em formato válido > mais ocorrências da mesma placa
    (consenso entre variantes) > menos correções > maior confiança do OCR.
    """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_text_utils.py -v`
Expected: PASS (todos, inclusive `test_menos_correcoes_vence_confianca` e `test_confianca_desempata_leituras_equivalentes` — ver nota abaixo).

> Nota de regressão (verificada): `test_menos_correcoes_vence_confianca` (`ABC1234` vs `48C1234`) passa porque **ambas extraem `ABC1234`** → n_ocorrencias=2 empata → desempata por `-correcoes` (0 vence 2). `test_confianca_desempata` (`ABC1234` vs `XYZ0A11`, placas distintas) passa porque n_ocorrencias=1 empata → confiança decide.

- [ ] **Step 5: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: 50 passed (47 originais + 1 Task 1 + 2 Task 3; os 2 testes reescritos na Task 2 não mudam a contagem).

- [ ] **Step 6: Commit**

```bash
git add src/core/text_utils.py tests/test_text_utils.py
git commit -m "fix(text): consenso entre variantes em escolher_leitura"
```

---

### Task 4: Gate no harness (43 imagens)

Não é código — é a medição que decide se a Fase 1.5 fecha. Roda o pipeline completo no container com as 3 mudanças e compara ao baseline.

**Files:** nenhum (execução + `eval/results/` copiado de volta).

- [ ] **Step 1: Subir containers (Redis primeiro)**

```bash
docker start parking_redis && sleep 3 && docker start vc_worker_portaria && sleep 5
```

- [ ] **Step 2: Copiar código atual + dataset e rodar**

```bash
cd /home/felipe/Projetos/VisionCore/vc-worker-portaria
docker exec vc_worker_portaria mkdir -p /app/eval /app/dataset
docker cp src/. vc_worker_portaria:/app/src/
docker cp eval/. vc_worker_portaria:/app/eval/
docker cp ../dataset/. vc_worker_portaria:/app/dataset/
docker exec -w /app vc_worker_portaria python eval/run_eval.py   # ~4-6 min CPU
docker cp "vc_worker_portaria:/app/eval/results/." eval/results/
docker stop vc_worker_portaria parking_redis
```

- [ ] **Step 3: Avaliar contra o gate**

Comparar o RESUMO ao baseline:
- estrita ≥ **37.2%** ✔/✗
- CER ≤ **29.6%** ✔/✗
- `revisar` < **15** ✔/✗
- sucessos errados ≤ **7** ✔/✗

Se `revisar` não caiu, aplicar o escalonamento da Task 2 (MIN_CHARS_CANDIDATA 5→3), recomitar e re-rodar. Se um sucesso-errado novo surgir em `29.jpg`/`37.jpg`, registrar e seguir (fora de escopo).

- [ ] **Step 4: Commit dos resultados**

```bash
git add eval/results/
git commit -m "test(eval): resultado da Fase 1.5 sobre 43 imagens"
```

---

## Self-Review

- **Cobertura do spec:** Fix 1 → Task 2; Fix 2 → Task 3; instrumentação sucessos-errados → Task 1; gate/baseline 43 → Task 4. ✔
- **Placeholders:** nenhum — todo passo com código/comando concreto. ✔
- **Consistência de tipos:** `imprimir_relatorio -> dict` com nova chave `n_sucessos_errados`; `ler_texto` e `escolher_leitura` mantêm assinaturas; `Extracao.placa/.valida/.correcoes` e `Decisao(...)` usados como definidos em `text_utils`. ✔
- **Contagem de testes:** 47 → 48 (Task 1) → 50 (Task 3); Task 2 reescreve 2 testes sem mudar contagem. ✔
