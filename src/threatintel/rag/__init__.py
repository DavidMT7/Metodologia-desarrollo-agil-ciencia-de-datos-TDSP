# -*- coding: utf-8 -*-
"""Retrieval, compuerta y generación anclada para el asistente CTI."""

from threatintel.rag.lineage_gate import LineageError, assert_phase3_lineage
from threatintel.rag.responder import load_responder, responder

__all__ = [
    "assert_phase3_lineage",
    "LineageError",
    "load_responder",
    "responder",
]
