"""Harness de avaliação offline do pipeline de extração de placas.

Roda o pipeline atual (YOLOv8 ONNX → pré-processamento → EasyOCR →
corrigir_placa) sobre o dataset local com ground truth anotado e reporta
acurácia estrita, CER e distribuição de erros — sem Redis/MinIO/API.

Uso (dentro do container do worker, a partir de /app):
    python eval/run_eval.py [--images-dir dataset] [--gt eval/ground_truth.csv]

Uso local (requer requirements.txt instalado e modelos em models/):
    python eval/run_eval.py --images-dir ../dataset
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.text_utils import corrigir_placa  # noqa: E402


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + (ca != cb)))
        anterior = atual
    return anterior[-1]


def carregar_ground_truth(caminho: str) -> dict[str, str]:
    gt = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            arquivo, placa = linha.split(",")
            gt[arquivo.strip()] = placa.strip().upper()
    return gt


def avaliar(detector, ocr_reader, gt: dict[str, str], images_dir: str) -> list[dict]:
    resultados = []
    for arquivo, placa_real in sorted(gt.items(), key=lambda kv: kv[0]):
        caminho = os.path.join(images_dir, arquivo)
        img = cv2.imread(caminho)
        if img is None:
            resultados.append({"arquivo": arquivo, "gt": placa_real, "erro": "imagem nao encontrada"})
            continue

        crop, conf_yolo = detector.detectar(img)
        if crop is None:
            resultados.append({
                "arquivo": arquivo, "gt": placa_real, "lido": None,
                "conf_yolo": 0.0, "status": "sem_deteccao", "dist": len(placa_real),
            })
            continue

        texto_bruto, _ = ocr_reader.ler_texto(crop)
        placa = corrigir_placa(texto_bruto)
        status = "sucesso" if len(placa) == 7 else "filtrado"
        resultados.append({
            "arquivo": arquivo,
            "gt": placa_real,
            "bruto": texto_bruto,
            "lido": placa,
            "conf_yolo": round(float(conf_yolo), 3),
            "status": status,
            "dist": levenshtein(placa, placa_real),
        })
    return resultados


def imprimir_relatorio(resultados: list[dict]) -> dict:
    validos = [r for r in resultados if "erro" not in r]
    n = len(validos)
    acertos = [r for r in validos if r["lido"] == r["gt"]]
    sucessos = [r for r in validos if r["status"] == "sucesso"]
    acertos_sucesso = [r for r in sucessos if r["lido"] == r["gt"]]
    cer = sum(r["dist"] / max(len(r["gt"]), 1) for r in validos) / n if n else 0.0

    dist_erros = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for r in validos:
        d = r["dist"]
        dist_erros["0" if d == 0 else "1" if d == 1 else "2" if d == 2 else "3+"] += 1

    largura = max(len(r["arquivo"]) for r in validos) if validos else 10
    print(f"\n{'arquivo':<{largura}}  {'gt':<8} {'lido':<8} {'bruto':<12} {'conf':<6} {'dist':<4} status")
    print("-" * (largura + 50))
    for r in validos:
        marca = "OK " if r["lido"] == r["gt"] else "ERR"
        print(f"{r['arquivo']:<{largura}}  {r['gt']:<8} {str(r['lido']):<8} "
              f"{str(r.get('bruto', ''))[:12]:<12} {r['conf_yolo']:<6} {r['dist']:<4} {marca} {r['status']}")

    resumo = {
        "n_imagens": n,
        "acuracia_estrita": round(len(acertos) / n, 4) if n else 0.0,
        "acuracia_entre_sucessos": round(len(acertos_sucesso) / len(sucessos), 4) if sucessos else 0.0,
        "n_sucessos": len(sucessos),
        "n_filtrados": sum(1 for r in validos if r["status"] == "filtrado"),
        "n_sem_deteccao": sum(1 for r in validos if r["status"] == "sem_deteccao"),
        "cer_medio": round(cer, 4),
        "distribuicao_erros": dist_erros,
    }
    print("\n=== RESUMO ===")
    print(f"Imagens avaliadas:        {resumo['n_imagens']}")
    print(f"Acurácia estrita (geral): {resumo['acuracia_estrita']:.1%}  ({len(acertos)}/{n})")
    print(f"Acurácia entre sucessos:  {resumo['acuracia_entre_sucessos']:.1%}  ({len(acertos_sucesso)}/{len(sucessos)})")
    print(f"Filtrados: {resumo['n_filtrados']}  ·  Sem detecção: {resumo['n_sem_deteccao']}")
    print(f"CER médio (Levenshtein/len): {resumo['cer_medio']:.1%}")
    print(f"Distribuição de erros por placa: {dist_erros}")
    return resumo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default="dataset")
    parser.add_argument("--gt", default="eval/ground_truth.csv")
    parser.add_argument("--model", default="models/modelo_placas.onnx")
    parser.add_argument("--out", default="eval/results", help="pasta para salvar o JSON do run")
    args = parser.parse_args()

    import easyocr  # import tardio: pesado

    from src.config import USE_GPU
    from src.services.ia_service import ONNXDetector
    from src.services.ocr_service import EasyOCRReader

    detector = ONNXDetector(args.model)
    leitor = easyocr.Reader(["pt", "en"], gpu=USE_GPU, model_storage_directory="models")
    ocr_reader = EasyOCRReader(leitor)

    gt = carregar_ground_truth(args.gt)
    print(f"Avaliando {len(gt)} imagens de {args.images_dir} contra {args.gt}")
    resultados = avaliar(detector, ocr_reader, gt, args.images_dir)
    resumo = imprimir_relatorio(resultados)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destino = os.path.join(args.out, f"eval_{stamp}.json")
    with open(destino, "w", encoding="utf-8") as f:
        json.dump({"resumo": resumo, "resultados": resultados}, f, ensure_ascii=False, indent=2)
    print(f"\nResultado salvo em {destino}")


if __name__ == "__main__":
    main()
