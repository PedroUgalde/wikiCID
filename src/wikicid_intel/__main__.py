"""CLI: python -m wikicid_intel --excel path [--query texto]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wikicid_intel.config import DEFAULT_EXCEL_PATH, DEFAULT_SHEET_NAME
from wikicid_intel.services.company_index import load_index


def main() -> None:
    p = argparse.ArgumentParser(description="WikiCID Intel CLI")
    p.add_argument("--excel", type=Path, default=DEFAULT_EXCEL_PATH)
    p.add_argument("--sheet", type=str, default=DEFAULT_SHEET_NAME)
    p.add_argument("--query", type=str, default="", help="Si se omite, solo muestra stats del índice")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    idx = load_index(args.excel, sheet_name=args.sheet)
    print(
        json.dumps(
            {
                "excel": str(args.excel.resolve()),
                "sheet": args.sheet,
                "rows": len(idx.df),
                "dedup": {
                    "before": idx.df.attrs.get("rows_before_dedup"),
                    "after": idx.df.attrs.get("rows_after_dedup"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.query.strip():
        hits = idx.search(args.query.strip(), top_k=args.top_k)
        out = [
            {
                "rank": h.rank,
                "empresa": h.empresa,
                "similitud_semantica": round(h.similitud_semantica, 4),
                "industria": round(h.score_industria, 1),
                "impacto": round(h.score_impacto, 1),
                "madurez": round(h.score_madurez, 1),
                "radar_total": round(h.radar_total, 1),
                "combinado": round(h.score_combinado, 4),
                "cluster": h.cluster,
            }
            for h in hits
        ]
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
