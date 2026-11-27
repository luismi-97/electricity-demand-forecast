"""Turn the result CSVs into Markdown and keep the README results block in sync.

    python -m edf.report            # writes reports/RESULTS.md and refreshes the block between
                                    # <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md

Nothing here is typed by hand: every number comes from the CSVs written by notebooks 04 (validation) and 05 (final
test). If the final test has not been run yet, the block says so.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from edf.config import ROOT

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
SETS = {
    "final": ("Final test (locked period, evaluated once)", "final_test.csv", "final_test_paired.csv",
              "final_test_intervals.csv"),
    "validation": ("Validation 2023-2024 (used for model development)", "week4_validation.csv",
                   "week4_paired_tests.csv", "week4_intervals.csv"),
}
NAMES = {"blend_lgbm_nn": "Blend (LightGBM + NN)", "lgbm_+lags": "LightGBM", "nn_mlp": "Neural network",
         "lgbm_quantile": "LightGBM (quantile, q50)", "official_tso": "Official TSO forecast",
         "naive_168h": "Naive (same hour last week)"}


def _read(path: Path):
    return pd.read_csv(path, index_col=0) if path.exists() else None


def load_set(reports: Path, key: str) -> dict | None:
    _, summ, paired, iv = SETS[key]
    s = _read(reports / summ)
    return None if s is None else {"summary": s, "paired": _read(reports / paired), "intervals": _read(reports / iv)}


def md_table(df: pd.DataFrame, formats: dict | None = None) -> str:
    formats = formats or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [formats.get(c, "{:,.0f}").format(row[c]) if pd.notna(row[c]) else "" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def headline(summary: pd.DataFrame, paired: pd.DataFrame | None) -> list[str]:
    out = []
    best = "blend_lgbm_nn" if "blend_lgbm_nn" in summary.index else summary["MAE"].idxmin()
    for ref, label in (("official_tso", "the official TSO forecast"), ("naive_168h", "the weekly naive baseline")):
        if ref in summary.index and best in summary.index and best != ref:
            red = (1 - summary.loc[best, "MAE"] / summary.loc[ref, "MAE"]) * 100
            line = (f"MAE of **{NAMES.get(best, best)}**: {summary.loc[best, 'MAE']:,.0f} MWh vs. "
                    f"{summary.loc[ref, 'MAE']:,.0f} MWh for {label} "
                    f"({abs(red):.1f} % {'lower' if red >= 0 else 'higher'} error)")
            key = f"{best} vs {ref}"
            if paired is not None and key in paired.index:
                p = paired.loc[key]
                line += (f"; paired bootstrap over days: difference {p['diff_a_minus_b']:+,.0f} MWh "
                         f"(95 % CI {p['ci95_low']:+,.0f} to {p['ci95_high']:+,.0f}), lower error on "
                         f"{p['share_days_a_better'] * 100:.0f} % of days")
            out.append(line)
    return out


def section(reports: Path, key: str) -> str | None:
    d = load_set(reports, key)
    if d is None:
        return None
    s = d["summary"].rename(index=NAMES)
    s.index.name = "Model"
    fmt = {"n": "{:,.0f}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE_%": "{:.2f}", "bias": "{:+,.0f}", "MASE": "{:.2f}"}
    parts = [f"### {SETS[key][0]}", ""]
    parts += [f"- {h}" for h in headline(d["summary"], d["paired"])]
    parts += ["", "Point forecasts (MAE, RMSE and bias in MWh; MAPE in %; MASE scaled by the weekly naive):", "",
              md_table(s, fmt), ""]
    if d["intervals"] is not None:
        iv = d["intervals"].copy()
        iv.index.name = "Interval"
        parts += ["90 % prediction intervals (nominal coverage 0.90):", "",
                  md_table(iv, {"n": "{:,.0f}", "nominal": "{:.2f}", "coverage": "{:.3f}", "mean_width": "{:,.0f}",
                                "width_%_of_load": "{:.1f}"}), ""]
    return "\n".join(parts)


def build_results_md(reports: Path) -> str:
    final, val = section(reports, "final"), section(reports, "validation")
    parts = ["## Results", ""]
    parts.append(final if final else "### Final test\n\n_Not evaluated yet. It is run once, after freezing the "
                 "configuration: `make final-test`, then `make report`._\n")
    if val:
        parts.append(val)
    parts += ["> Caveats: the models see the **observed** weather of the target day (the official forecast was made "
              "with weather forecasts) and use load up to 12:00 on D-1, which may differ from the TSOs' information "
              "cutoff. Read the comparison as 'under this protocol', not as a like-for-like contest.", ""]
    return "\n".join(parts)


def update_readme(readme: Path, block: str) -> bool:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    new = re.sub(f"{re.escape(START)}.*?{re.escape(END)}", lambda m: f"{START}\n{block}\n{END}", text, flags=re.S)
    readme.write_text(new, encoding="utf-8")
    return True


def main() -> None:
    reports = ROOT / "reports"
    md = build_results_md(reports)
    reports.mkdir(exist_ok=True)
    (reports / "RESULTS.md").write_text(md, encoding="utf-8")
    ok = update_readme(ROOT / "README.md", md)
    print(f"Wrote {reports / 'RESULTS.md'}; README block {'updated' if ok else 'markers not found (not changed)'}")


if __name__ == "__main__":
    main()
