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
