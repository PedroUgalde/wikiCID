from __future__ import annotations

from pathlib import Path

import pandas as pd

from wikicid_intel.config import DEFAULT_SHEET_NAME, SEMANTIC_COLUMNS


def load_empresas(
    excel_path: str | Path,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    path = Path(excel_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el Excel: {path.resolve()}")

    sheet = sheet_name or DEFAULT_SHEET_NAME
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    missing = [c for c in SEMANTIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas en el Excel (hoja '{sheet}'): {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    return df


def semantic_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Solo columnas útiles para texto + ID si existe."""
    cols = [c for c in ("ID", "Fecha", *SEMANTIC_COLUMNS) if c in df.columns]
    return df[cols].copy()
