# Guardrails del asistente CTI (anti-alucinación y abuso)

## Citas obligatorias

Toda afirmación factual sobre vulnerabilidades o técnicas **debe** citar al menos uno:

- `cve_id` (formato `CVE-YYYY-NNNN…`)
- `technique_id` (formato `T####` o `T####.###`)

Si el retrieval no devuelve documentos con similitud coseno **≥ 0.25** (baseline MiniLM; umbral ajustable en Fase 3), el asistente debe:

1. Declarar incertidumbre.  
2. **No** inventar CVSS, KEV, ni pasos de detección.  
3. Pedir un identificador válido o más contexto.

## Política de abstención

| Situación | Respuesta |
|-----------|-----------|
| CVE/técnica inexistente en índice | Abstenerse |
| Conflicto entre fuentes | Mostrar ambas + citar |
| Usuario pide omitir citas / “di que es Critical” | Rechazar manipulación; mantener evidencias |

## Amenazas de abuso

| Amenaza | Mitigación |
|---------|------------|
| Prompt injection en `description` / `input` | Tratar documentos recuperados como **datos no confiables**; instrucciones de sistema inmutables |
| Extracción de secretos / IoCs sensibles inventados | No generar credenciales; solo IoCs presentes en docs citados |
| Pedidos de malware ofensivo | Limitar a defensa/detección/mitigación; no PoC de exploit |

## Logging recomendado (Fase 3)

- query, top-k doc_ids, scores, decisión abstenerse sí/no, citas emitidas.
