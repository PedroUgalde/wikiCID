from __future__ import annotations

import sys
from pathlib import Path

# Permite ejecutar con `streamlit run src/wikicid_intel/ui/streamlit_app.py` desde CARSO
_SRC_PARENT = Path(__file__).resolve().parents[2]
if str(_SRC_PARENT) not in sys.path:
    sys.path.insert(0, str(_SRC_PARENT))

import streamlit as st

from wikicid_intel.config import DEFAULT_EXCEL_PATH, DEFAULT_SHEET_NAME, SEMANTIC_COLUMNS
from wikicid_intel.services.company_index import SearchHit, load_index


@st.cache_resource(show_spinner="Construyendo índice semántico y Radar…")
def cached_index(excel_path: str, sheet_name: str):
    return load_index(excel_path, sheet_name=sheet_name or None)


def hit_to_row(h: SearchHit) -> dict:
    return {
        "#": h.rank,
        "Empresa": h.empresa,
        "Sim. semántica": round(h.similitud_semantica, 4),
        "Industria (35%)": round(h.score_industria, 1),
        "Impacto (35%)": round(h.score_impacto, 1),
        "Madurez (30%)": round(h.score_madurez, 1),
        "Radar total": round(h.radar_total, 1),
        "Ranking combinado": round(h.score_combinado, 4),
        "Cluster": h.cluster,
        "Mercados": h.mercados[:120] + ("…" if len(h.mercados) > 120 else ""),
        "Web": h.pagina_web,
    }


def main() -> None:
    st.set_page_config(page_title="WikiCID Intel", layout="wide")
    st.title("WikiCID · Buscador semántico de empresas")
    st.caption(
        "Consulta en lenguaje natural. El ranking mezcla similitud con tu consulta y el Radar "
        "(industria, impacto, madurez de datos)."
    )

    default_path = str(DEFAULT_EXCEL_PATH.resolve())
    col_a, col_b = st.columns(2)
    with col_a:
        excel_path = st.text_input("Ruta a reporte_47.xlsx", value=default_path)
    with col_b:
        sheet = st.text_input("Hoja", value=DEFAULT_SHEET_NAME)

    if not Path(excel_path).is_file():
        st.error(f"No existe el archivo: {excel_path}")
        st.info(
            "Coloca `reporte_47.xlsx` en la raíz de CARSO o indica la ruta completa. "
            "Variables de entorno: `WIKICID_EXCEL`, `WIKICID_SHEET`."
        )
        return

    with st.expander("Columnas usadas para el texto semántico"):
        st.write(", ".join(SEMANTIC_COLUMNS))

    try:
        index = cached_index(excel_path, sheet)
    except Exception as e:
        st.exception(e)
        return

    st.success(
        f"Índice listo: **{index.df.attrs.get('rows_after_dedup', len(index.df))}** empresas "
        f"(antes de deduplicar: {index.df.attrs.get('rows_before_dedup', '—')})."
    )

    q = st.text_input(
        "Búsqueda semántica",
        placeholder="Ej.: optimización de call center con IA en retail",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        top_k = st.slider("Resultados", 5, 40, 15)
    with c2:
        w_sem = st.slider("Peso similitud", 0.0, 1.0, 0.55, 0.05)
    with c3:
        w_rad = st.slider("Peso Radar", 0.0, 1.0, 0.45, 0.05)

    if st.button("Buscar", type="primary") and q.strip():
        hits = index.search(q.strip(), top_k=top_k, weight_semantic=w_sem, weight_radar=w_rad)
        st.subheader("Ranking")
        st.dataframe([hit_to_row(h) for h in hits], use_container_width=True, hide_index=True)
        st.subheader("Detalle (primer resultado)")
        if hits:
            h0 = hits[0]
            st.markdown(f"**{h0.empresa}** · cluster `{h0.cluster}`")
            st.write(h0.descripcion)
    elif q.strip():
        st.info('Pulsa "Buscar" para ejecutar la consulta.')


if __name__ == "__main__":
    main()
