# WikiCID Intel (CARSO)

- **Datos**: `reporte_47.xlsx` en la raíz del workspace; hoja por defecto `Empresas`.
- **Entrada semántica**: columnas útiles `Empresa`, `Descripción`, `Mercados`, `Tamaño`, `Fondeo`, `Ingresos`, `Valuación`, `Oportunidades`, `Sede`, `Alianzas`, `Página Web`. Redes sociales se omiten del texto de embedding.
- **Embeddings**: modelo multilingüe `paraphrase-multilingual-MiniLM-L12-v2`. El texto para embedding usa limpieza ligera (mejor semántica); la huella con lematización/stem sirve para deduplicación.
- **Radar**: industria 35%, impacto 35%, madurez de datos 30%; la búsqueda combina similitud semántica con el Radar.
- **URLs**: `WIKICID_PROBE_WEB=0` omite peticiones HTTP al construir el índice (más rápido; solo puntúa host parseado).
- **UI**: desde CARSO, `streamlit run src/wikicid_intel/ui/streamlit_app.py` (tras `pip install -e .` o con `PYTHONPATH=src`).
