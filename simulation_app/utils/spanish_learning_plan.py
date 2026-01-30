"""
Spanish Learning Plan Generator
===============================
Builds a structured, output-focused learning plan based on learner needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any


INTENSITY_PRESETS = {
    "Light": {
        "daily_output_prompts": 2,
        "weekly_domains": 2,
        "review_cycle_days": [1, 3, 7],
    },
    "Standard": {
        "daily_output_prompts": 4,
        "weekly_domains": 3,
        "review_cycle_days": [1, 3, 7, 14],
    },
    "Intensive": {
        "daily_output_prompts": 6,
        "weekly_domains": 4,
        "review_cycle_days": [1, 2, 4, 7, 14],
    },
}

DOMAIN_SUGGESTIONS = {
    "Cultura y sociedad": [
        "costumbres regionales", "normas sociales", "humor y sarcasmo",
    ],
    "Salud y bienestar": [
        "síntomas", "tratamientos", "hábitos", "salud mental",
    ],
    "Trabajo y carrera": [
        "reuniones", "negociación", "evaluaciones", "liderazgo",
    ],
    "Vida cotidiana avanzada": [
        "trámites", "alquileres", "servicios", "reclamos",
    ],
    "Tecnología aplicada": [
        "automatización", "privacidad", "herramientas digitales",
    ],
    "Creatividad y arte": [
        "crítica", "estilos", "procesos", "técnicas",
    ],
    "Economía personal": [
        "inversiones", "deudas", "presupuestos", "ahorro",
    ],
}

VERB_CONTRASTS = [
    (
        "apurar", "acelerar",
        "'Apurar' implica urgencia o presión social; 'acelerar' es más mecánico o técnico.",
        "Apura el trámite si no quieres perder el turno. / El sistema acelera el proceso de registro.",
    ),
    (
        "plantear", "proponer",
        "'Plantear' introduce un tema o problema; 'proponer' sugiere una solución concreta.",
        "Planteó una duda clave en la reunión. / Propuso un plan con fechas y costos.",
    ),
    (
        "matizar", "suavizar",
        "'Matizar' agrega precisión; 'suavizar' reduce intensidad o impacto emocional.",
        "Matizó su crítica con datos. / Suavizó el tono para evitar conflicto.",
    ),
    (
        "asumir", "presumir",
        "'Asumir' puede ser aceptar o suponer; 'presumir' es suponer con menos evidencia.",
        "Asumimos la responsabilidad del error. / Presumí que llegarías tarde.",
    ),
    (
        "desencadenar", "provocar",
        "'Desencadenar' sugiere una cadena de eventos; 'provocar' es más directo.",
        "El comentario desencadenó una discusión larga. / La decisión provocó malestar.",
    ),
    (
        "esbozar", "detallar",
        "'Esbozar' es dar una idea general; 'detallar' es entrar en precisión.",
        "Esbozó la estrategia en cinco minutos. / Detalló cada paso del proceso.",
    ),
]

GRAMMAR_FOCUS_LIBRARY = {
    "Género y concordancia": [
        "Sustantivo + adjetivo en género y número", "Determinantes con profesiones y roles",
    ],
    "Tiempos verbales": [
        "Pretérito perfecto vs. imperfecto", "Condicional para hipótesis reales", "Subjuntivo en opiniones",
    ],
    "Conectores avanzados": [
        "sin embargo / aun así / pese a", "además / asimismo / de hecho",
    ],
    "Pronombres y clíticos": [
        "se lo / se la / se los", "leísmo y laísmo (regional)",
    ],
}

OUTPUT_PROMPTS = {
    "Escritura": [
        "Redacta un mensaje profesional explicando un retraso y ofreciendo alternativas.",
        "Escribe una mini-crónica sobre un problema cotidiano y cómo lo resolviste.",
        "Argumenta a favor o en contra de una decisión laboral reciente usando conectores avanzados.",
    ],
    "Habla": [
        "Describe en voz alta una experiencia reciente usando verbos de precisión (p. ej., 'matizar').",
        "Imagina que estás negociando un plazo: explica tu postura y cede en un punto.",
        "Cuenta una historia breve con un giro inesperado y usa el pasado correctamente.",
    ],
    "Diálogo": [
        "Simula una conversación difícil con un compañero y practica el tono adecuado.",
        "Role-play: cliente exigente y respuesta diplomática con verbos matizados.",
        "Debate rápido sobre un cambio de política en el trabajo.",
    ],
}


@dataclass
class SpanishLearningProfile:
    comfort_topics: str
    target_domains: List[str]
    grammar_focus: List[str]
    output_modes: List[str]
    intensity: str
    learner_notes: str = ""


def _format_list(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_spanish_learning_plan(profile: Dict[str, Any]) -> str:
    intensity = profile.get("intensity", "Standard")
    preset = INTENSITY_PRESETS.get(intensity, INTENSITY_PRESETS["Standard"])
    target_domains = profile.get("target_domains", [])
    grammar_focus = profile.get("grammar_focus", [])
    output_modes = profile.get("output_modes", [])

    domain_sections = []
    for domain in target_domains[: preset["weekly_domains"]]:
        suggestions = DOMAIN_SUGGESTIONS.get(domain, [])
        if suggestions:
            domain_sections.append(f"### {domain}\n{_format_list(suggestions)}")
        else:
            domain_sections.append(f"### {domain}\n- Explora vocabulario específico y ejemplos reales.")

    grammar_sections = []
    for focus in grammar_focus:
        drills = GRAMMAR_FOCUS_LIBRARY.get(focus, ["Ejercicios de aplicación en contexto."])
        grammar_sections.append(f"### {focus}\n{_format_list(drills)}")

    prompts = []
    for mode in output_modes:
        prompts.extend(OUTPUT_PROMPTS.get(mode, []))

    prompts = prompts[: preset["daily_output_prompts"]]

    review_cycle = ", ".join(str(day) for day in preset["review_cycle_days"])

    verb_contrast_section = "\n".join(
        f"- **{a} vs. {b}**: {note} Ejemplo: {example}"
        for a, b, note, example in VERB_CONTRASTS
    )

    notes = profile.get("learner_notes", "").strip()
    notes_block = f"\n**Notas del estudiante:** {notes}" if notes else ""

    return (
        "# Plan personalizado de español\n"
        "\n"
        "## Perfil de entrada\n"
        f"- Temas de confort actuales: {profile.get('comfort_topics', 'N/A')}\n"
        f"- Intensidad: {intensity}\n"
        f"- Modos de producción: {', '.join(output_modes) if output_modes else 'N/A'}\n"
        f"- Enfoque gramatical: {', '.join(grammar_focus) if grammar_focus else 'N/A'}\n"
        f"- Dominios nuevos: {', '.join(target_domains) if target_domains else 'N/A'}\n"
        f"{notes_block}\n"
        "\n"
        "## 1) Expansión de vocabulario fuera de la zona de confort\n"
        "- Exposición semanal a dominios nuevos con vocabulario contextualizado.\n"
        "- Cada dominio incluye mini-retos de uso activo y ejemplos auténticos.\n"
        "\n"
        f"{chr(10).join(domain_sections) if domain_sections else '- Selecciona dominios para generar módulos.'}\n"
        "\n"
        "## 2) Verbos de acción avanzados y precisión\n"
        "- Selección de verbos menos comunes con contraste de matiz.\n"
        "- Práctica de elección del verbo más adecuado según contexto.\n"
        f"\n{verb_contrast_section}\n"
        "\n"
        "## 3) Refuerzo gramatical automático\n"
        "- Corrección rápida con explicación mínima y ejemplos de refuerzo.\n"
        "- Repetición distribuida en múltiples contextos.\n"
        f"\n{chr(10).join(grammar_sections) if grammar_sections else '- Selecciona focos gramaticales para ejercicios.'}\n"
        "\n"
        "## 4) Producción activa (hablar y escribir)\n"
        "- Prompts diarios para producir con vocabulario y gramática focalizada.\n"
        f"- Ciclo de revisión espaciada: días {review_cycle}.\n"
        f"\n{_format_list(prompts) if prompts else '- Selecciona modos de salida para generar prompts.'}\n"
        "\n"
        "## 5) Personalización y seguimiento\n"
        "- Registra errores recurrentes y reintroduce el contenido en la siguiente semana.\n"
        "- Ajusta el nivel de dificultad cuando los errores bajen al 10% o menos.\n"
        "\n"
        "**Plantilla rápida de seguimiento:**\n"
        "| Fecha | Error | Corrección | Ejemplo propio |\n"
        "| --- | --- | --- | --- |\n"
        "|  |  |  |  |\n"
    )
