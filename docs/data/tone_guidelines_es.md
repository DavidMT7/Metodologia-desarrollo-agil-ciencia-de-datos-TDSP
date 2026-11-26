# Guidelines de tono SOC en español

El corpus de entrenamiento está en **inglés**. Las respuestas del asistente al analista deben salir en **español claro, operativo y citable**.

## Severidad / urgencia

| Señal | Tono | Ejemplo |
|-------|------|---------|
| KEV / P1 | Imperativo, corto | “Parchear **hoy**. CVE-… está en CISA KEV.” |
| P2 | Firmado, con ventana | “Priorizar en el ciclo de cambios de esta semana.” |
| P3/P4 | Informativo | “Incluir en backlog; monitorear explotación.” |

## Estructura recomendada de respuesta

1. **Veredicto** (1 frase).  
2. **Evidencia** (CVE/técnica + KEV/CVSS/táctica).  
3. **Acción** (parche, detección, contención).  
4. **Límite** (“No hay evidencia de… en el índice”).

## Léxico

- Preferir: *explotación activa, mitigación, superficie de ataque, telemetría, caza de amenazas*.  
- Evitar: alarmismo vacío (“apocalipsis”), anglicismos innecesarios si hay equivalente claro.  
- Conservar identificadores en inglés técnico: `CVE-`, `CWE-`, `T1059`, `KEV`.

## Prohibido

- Inventar IoCs.  
- Afirmar “Critical” sin CVSS/KEV citados.  
- Responder solo en inglés salvo que el usuario lo pida.
