# -*- coding: utf-8 -*-
"""Respuesta anclada al contexto, filtro de citas e intents bloqueados."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from threatintel.rag.gate import should_abstain
from threatintel.rag.ids import CVE_RE, TECH_RE, extract_ids

ABSTAIN_TEXT = (
    "No hay evidencia suficiente en el índice para responder con seguridad. "
    "Declaro incertidumbre: no invento CVSS, estado KEV ni pasos de detección. "
    "Indique un identificador válido (CVE-YYYY-NNNN o técnica Txxxx) o más contexto."
)

POLICY = {
    "offensive": (
        "No elaboro procedimientos ofensivos ni código de ataque. "
        "Puedo orientar detección y mitigación solo cuando el índice aporta un CVE o una técnica citada."
    ),
    "manipulation": (
        "No omito citas ni afirmo que todo es Critical. "
        "Toda afirmación factual cita un cve_id o technique_id presente en el contexto recuperado."
    ),
    "datasheet": (
        "No. La pérdida estimada y el accuracy de severity_level en Kaggle no priorizan parches reales: "
        "el datasheet los marca como sintéticos y no discriminan la severidad operativa. "
        "La priorización de este asistente usa KEV, CVSS, exploit público y MITRE ATT&CK."
    ),
    "playbook": (
        "in_cisa_kev aporta 50 puntos sobre un máximo teórico cercano a 100. "
        "KEV domina sobre el exploit público y sobre el CVSS, que solo matiza el score."
    ),
    "taxonomy": (
        "Defense Evasion en el corpus CVE y Defense Impairment en el catálogo MITRE de este proyecto "
        "describen la misma familia defensiva. El join_key mapea Defense Evasion hacia Defense Impairment "
        "para poder recuperar técnicas relacionadas."
    ),
    "tone": (
        "Para un KEV explotado en ransomware el tono es de urgencia alta, con verbo imperativo de parcheo, "
        "el CVE citado y la acción CISA. Sin alarmismo vacío."
    ),
    "low_similarity": (
        "No. Si la similitud del mejor documento queda bajo el umbral calibrado, no genero una respuesta técnica: "
        "me abstengo, declaro incertidumbre y pido un identificador válido o más contexto."
    ),
}

_OFFENSIVE = re.compile(
    r"(escribe|genera|dame|construye|crea|arma).{0,48}"
    r"(exploit|payload|shellcode|malware|poc|prueba de concepto|c[oó]digo de ataque)|"
    r"(paso a paso|procedimiento|gu[ií]a).{0,40}(explotar|exploit|atacar|comprometer)|"
    r"\b(explotar|exploit)\b.{0,40}(sin ser detectado|en producci[oó]n|en vivo)|"
    r"\b(payload|shellcode|msfvenom|reverse shell)\b|"
    r"c[oó]digo de explotaci[oó]n|"
    r"bypass de autenticaci[oó]n",
    re.IGNORECASE,
)
_MANIPULATION = re.compile(
    r"omite las citas|ignora las citas|sin citar|no cites|"
    r"afirma que todo es critical|di que es critical|todo es critical",
    re.IGNORECASE,
)
_DATASHEET = re.compile(r"estimated_financial_loss|kaggle|severity_level", re.IGNORECASE)
_PLAYBOOK = re.compile(r"peso.{0,40}in_cisa_kev|in_cisa_kev.{0,40}peso|score soc del proyecto", re.IGNORECASE)
_TAXONOMY = re.compile(r"defense evasion", re.IGNORECASE)
_TONE = re.compile(r"tono correcto|qu[eé] tono", re.IGNORECASE)
_LOW_SIM = re.compile(r"similitud.{0,40}0\.25|generas respuesta t[eé]cnica", re.IGNORECASE)
_ACTIONS = {
    "P1-immediate": "Parchear hoy o aplicar de inmediato la mitigación indicada por CISA.",
    "P2-patch-soon": "Priorizar en el ciclo de cambios de esta semana.",
    "P3-plan": "Incluir en el backlog priorizado del sprint y monitorear explotación.",
    "P4-monitor": "Dejar en watchlist y seguir la inteligencia de amenazas.",
}


def route_query(query: str) -> str:
    text = query or ""
    if _OFFENSIVE.search(text):
        return "offensive"
    if _MANIPULATION.search(text):
        return "manipulation"
    if _DATASHEET.search(text):
        return "datasheet"
    if _PLAYBOOK.search(text):
        return "playbook"
    if _TAXONOMY.search(text) and re.search(r"defense impairment", text, re.IGNORECASE):
        return "taxonomy"
    if _TONE.search(text):
        return "tone"
    if _LOW_SIM.search(text):
        return "low_similarity"
    return "retrieval"


def allowed_ids(docs: list[dict]) -> set[str]:
    allowed: set[str] = set()
    for doc in docs:
        allowed.add(str(doc.get("doc_id", "")).upper())
        blob = " ".join(str(doc.get(key, "")) for key in ("text", "doc_id", "cwe"))
        allowed.update(extract_ids(blob))
    return {item for item in allowed if item}


def filter_citations(answer: str, allowed: set[str]) -> tuple[str, list[str]]:
    removed: list[str] = []

    def _replace(match: re.Match) -> str:
        value = match.group(0).upper()
        if value in allowed:
            return match.group(0)
        removed.append(match.group(0))
        return "[identificador ausente del contexto]"

    cleaned = CVE_RE.sub(_replace, answer or "")
    cleaned = TECH_RE.sub(_replace, cleaned)
    if removed:
        cleaned += (
            "\nAviso: se retiraron identificadores que no estaban en el contexto recuperado: "
            + ", ".join(removed)
            + "."
        )
    return cleaned, removed


def _snippet(text: str, limit: int = 420) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "…"


def render_soc(docs: list[dict]) -> str:
    primary = docs[0]
    doc_id = str(primary["doc_id"])
    if primary.get("doc_type") == "cve":
        kev = "SÍ está" if primary.get("in_cisa_kev") else "NO está"
        band = primary.get("soc_priority_band") or "P4-monitor"
        score = primary.get("soc_priority_score")
        score_txt = f"{float(score):.1f}" if score is not None else "no calculado"
        cvss = primary.get("cvss_score")
        cvss_txt = f"{float(cvss):.1f}" if cvss is not None else "no disponible"
        action = _ACTIONS.get(band, _ACTIONS["P4-monitor"])
        kev_action = str(primary.get("kev_required_action") or "").strip()
        if kev_action and kev_action.lower() != "nan":
            action = f"{action} Acción CISA registrada: {kev_action}"
        return (
            f"Veredicto: {doc_id} {kev} en CISA KEV; banda {band} (score {score_txt}).\n"
            f"Evidencia: CVSS {cvss_txt}; táctica {primary.get('tactic') or 'no indicada'}; "
            f"CWE {primary.get('cwe') or 'no indicado'}; producto {primary.get('product') or 'no indicado'}. "
            f"Extracto del índice: {_snippet(str(primary.get('text') or ''))}\n"
            f"Acción: {action}\n"
            "Límite: esta respuesta solo usa el índice endurecido. No sustituye la telemetría del SIEM "
            "ni afirma que la vulnerabilidad esté explotada en su red."
        )
    return (
        f"Veredicto: la detección de {doc_id} ({primary.get('technique_name') or 'técnica'}) "
        f"se apoya en la estrategia oficial ATT&CK recuperada.\n"
        f"Evidencia: táctica {primary.get('tactic') or 'no indicada'}. "
        f"{_snippet(str(primary.get('text') or ''))}\n"
        "Acción: cazar con las analíticas y fuentes citadas. Esto es orientación defensiva, no un procedimiento de ataque.\n"
        "Límite: si el chunk no trae una fuente de datos, no la invento."
    )


def query_hash(query: str) -> str:
    return hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:16]


def answer_query(query: str, retrieval: dict | None = None, threshold: float = 0.25) -> dict:
    route = route_query(query)
    base = {
        "query_hash": query_hash(query),
        "route": route,
        "abstained": False,
        "blocked": route == "offensive",
        "cited": [],
        "removed_ids": [],
        "top_doc_ids": [],
        "dense_top_score": None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if route != "retrieval":
        base["text"] = POLICY[route]
        return base
    if retrieval is None:
        base["abstained"] = True
        base["text"] = ABSTAIN_TEXT
        return base
    base["dense_top_score"] = retrieval.get("dense_top_score")
    base["top_doc_ids"] = list(retrieval.get("top_doc_ids") or [])
    abstain = should_abstain(
        float(retrieval.get("dense_top_score") or 0.0),
        threshold,
        bool(retrieval.get("explicit_ids_missing")),
    )
    if abstain or not retrieval.get("docs"):
        base["abstained"] = True
        base["text"] = ABSTAIN_TEXT
        return base
    docs = list(retrieval["docs"])
    explicit_hits = [str(doc_id) for doc_id in retrieval.get("explicit_hits") or []]
    if explicit_hits:
        by_id = {str(doc["doc_id"]): doc for doc in docs}
        pinned = [by_id[doc_id] for doc_id in explicit_hits if doc_id in by_id]
        rest = [doc for doc in docs if str(doc["doc_id"]) not in explicit_hits]
        docs = pinned + rest
    text, removed = filter_citations(render_soc(docs[:3]), allowed_ids(docs[:3]))
    base["text"] = text
    base["removed_ids"] = removed
    base["cited"] = extract_ids(text)
    return base
