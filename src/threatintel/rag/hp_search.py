# -*- coding: utf-8 -*-
"""Decisión de modelado. El gold no entra."""
from __future__ import annotations


def choose_deployment(ref_2ep: float, rows: list[dict], margin: float = 0.01) -> dict:
    """Compara el barrido contra las dos épocas ya publicadas.

    Nunca propone pisar `biencoder_best`: el gold de esa corrida ya se abrió.
    """
    if not rows:
        raise ValueError("El barrido no tiene filas")
    one_epoch = [row for row in rows if int(row["epochs"]) == 1 and row.get("role") != "confirmacion"]
    best_one = max(one_epoch, key=lambda row: float(row["dev_mrr_without_id"]))
    challengers = [
        row
        for row in rows
        if int(row["epochs"]) >= 2 and float(row["dev_mrr_without_id"]) > float(ref_2ep) + margin
    ]
    challenger = max(challengers, key=lambda row: float(row["dev_mrr_without_id"])) if challengers else None
    if challenger is None:
        reason = (
            f"Ningún entrenamiento de 2 o más épocas supera por {margin:.2f} "
            f"al checkpoint publicado (MRR de dev sin identificador {ref_2ep:.3f}). "
            "biencoder_best se queda."
        )
    else:
        reason = (
            f"{challenger['tag']} llega a {float(challenger['dev_mrr_without_id']):.3f} en dev, "
            f"por encima de {ref_2ep:.3f}. Queda como candidato y no pisa biencoder_best: "
            "el gold ya se había abierto y no desempató esta búsqueda."
        )
    return {
        "overwrite_published": False,
        "best_one_epoch_tag": best_one["tag"],
        "best_one_epoch_mrr": float(best_one["dev_mrr_without_id"]),
        "challenger_tag": None if challenger is None else challenger["tag"],
        "reason": reason,
    }
