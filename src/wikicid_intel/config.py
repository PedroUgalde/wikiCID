from __future__ import annotations

import os
from pathlib import Path

# Raíz del proyecto = carpeta que contiene reporte_47.xlsx (CARSO)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_EXCEL_PATH = Path(os.environ.get("WIKICID_EXCEL", _PROJECT_ROOT / "reporte_47.xlsx"))
DEFAULT_SHEET_NAME = os.environ.get("WIKICID_SHEET", "Empresas")

SEMANTIC_COLUMNS = [
    "Empresa",
    "Descripción",
    "Mercados",
    "Tamaño",
    "Fondeo",
    "Ingresos",
    "Valuación",
    "Oportunidades",
    "Sede",
    "Alianzas",
    "Página Web",
]

EMBEDDING_MODEL = os.environ.get(
    "WIKICID_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Pesos Radar (deben sumar 1.0)
WEIGHT_INDUSTRY = 0.35
WEIGHT_IMPACT = 0.35
WEIGHT_MATURITY = 0.30

# Fusión búsqueda: similitud coseno [0,1] + radar normalizado [0,1]
WEIGHT_SEMANTIC_IN_RANK = 0.55
WEIGHT_RADAR_IN_RANK = 0.45

INDUSTRY_ANCHORS_ES = [
    "empresa del sector de telecomunicaciones y conectividad",
    "empresa del sector bancario y servicios financieros tradicionales",
    "empresa fintech y tecnología financiera",
    "empresa de salud y tecnología médica",
    "empresa de educación y edtech",
    "empresa de medios entretenimiento y streaming",
    "empresa de música y audio",
    "empresa de agricultura y agtech",
    "empresa de minería y recursos",
    "empresa de energía y utilities",
    "empresa de retail y tiendas físicas",
    "empresa de comercio electrónico y marketplaces",
]

IMPACT_ANCHORS_ES = [
    "solución que ayuda a vender más y aumentar ingresos",
    "mejora la experiencia del cliente y la satisfacción del usuario",
    "optimiza costos procesos y eficiencia operativa",
]

CACHE_DIR = _PROJECT_ROOT / ".wikicid_cache"
