import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT.parent / "DATASET ORIGINAL" / "DATASET_FINAL_HACKATHON_2026.parquet"
OUTPUT_PATH = ROOT / "outputs" / "flights_model_ready.parquet"
DOCS_PATH = ROOT / "docs"

leakage_cols = ["DEP_DELAY"]
redundant_cols = [
    "ORIGIN_CITY_NAME",
    "DEST_CITY_NAME",
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DEP_HOUR",
    "CRS_DEP_TIME",
    "distance_bin",
    "DIST_MET_KM",
]

winsor_cols = ["TEMP", "WIND_SPD", "PRECIP_1H", "DISTANCE"]

print(f"Loading: {DATASET_PATH}")
df = pd.read_parquet(DATASET_PATH)
initial_rows = len(df)
initial_cols = list(df.columns)

rule_stats = {}

present_leakage = [c for c in leakage_cols if c in df.columns]
df = df.drop(columns=present_leakage)
rule_stats["drop_leakage_cols"] = {
    "columns": present_leakage,
    "rows_affected": 0,
}

present_redundant = [c for c in redundant_cols if c in df.columns]
df = df.drop(columns=present_redundant)
rule_stats["drop_redundant_cols"] = {
    "columns": present_redundant,
    "rows_affected": 0,
}

rows_sched_fix = 0
if "sched_minute_of_day" in df.columns:
    rows_sched_fix = int((df["sched_minute_of_day"] == 1440).sum())
    df.loc[df["sched_minute_of_day"] == 1440, "sched_minute_of_day"] = 0
rule_stats["fix_sched_minute_of_day"] = {
    "rows_affected": rows_sched_fix,
    "details": "Set sched_minute_of_day 1440 to 0 (midnight).",
}

rows_precip_fix = 0
if "PRECIP_1H" in df.columns:
    rows_precip_fix = int((df["PRECIP_1H"] < 0).sum())
    df.loc[df["PRECIP_1H"] < 0, "PRECIP_1H"] = 0
rule_stats["fix_precip_negative"] = {
    "rows_affected": rows_precip_fix,
    "details": "Set negative PRECIP_1H to 0.",
}

winsor_thresholds = {}
winsor_counts = {}
for col in winsor_cols:
    if col in df.columns:
        low, high = df[col].quantile([0.01, 0.99])
        winsor_thresholds[col] = {"p01": float(low), "p99": float(high)}
        below = int((df[col] < low).sum())
        above = int((df[col] > high).sum())
        winsor_counts[col] = {"below": below, "above": above}
        df[col] = df[col].clip(lower=low, upper=high)

rule_stats["winsorize_outliers"] = {
    "columns": [c for c in winsor_cols if c in df.columns],
    "rows_affected": int(sum(v["below"] + v["above"] for v in winsor_counts.values())),
    "details": "Winsorization at 1% and 99% for selected numeric columns.",
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving: {OUTPUT_PATH}")
df.to_parquet(OUTPUT_PATH, index=False)

final_rows = len(df)
final_cols = list(df.columns)

DOCS_PATH.mkdir(parents=True, exist_ok=True)

rules_md = """# Reglas de limpieza

1. Eliminar variables con fuga de informacion: DEP_DELAY.
2. Eliminar redundancias: ORIGIN_CITY_NAME, DEST_CITY_NAME, YEAR, MONTH, DAY_OF_MONTH, DEP_HOUR, CRS_DEP_TIME, distance_bin, DIST_MET_KM.
3. Corregir dominios:
   - sched_minute_of_day: 1440 -> 0 (medianoche).
   - PRECIP_1H negativa -> 0.
4. Tratar outliers (winsorizacion 1%/99%) en TEMP, WIND_SPD, PRECIP_1H, DISTANCE para robustez del MVP.

Justificacion de outliers:
- En un MVP explicable se busca estabilidad frente a valores extremos no representativos sin eliminar registros.
"""
(DOCS_PATH / "reglas_limpieza.md").write_text(rules_md, encoding="utf-8")

resumen_md = """# Resumen Antes vs Despues

## Filas y columnas
- Filas iniciales: {initial_rows}
- Filas finales: {final_rows}
- Columnas iniciales: {initial_cols}
- Columnas finales: {final_cols}

## Columnas eliminadas
- Fuga: {leakage_cols}
- Redundancias: {redundant_cols}

## Correcciones de dominio
- sched_minute_of_day 1440->0: {rows_sched_fix} filas
- PRECIP_1H negativa->0: {rows_precip_fix} filas

## Outliers (winsorizacion p01/p99)
- Umbrales: {winsor_thresholds}
- Valores recortados: {winsor_counts}
""".format(
    initial_rows=initial_rows,
    final_rows=final_rows,
    initial_cols=len(initial_cols),
    final_cols=len(final_cols),
    leakage_cols=present_leakage,
    redundant_cols=present_redundant,
    rows_sched_fix=rows_sched_fix,
    rows_precip_fix=rows_precip_fix,
    winsor_thresholds=json.dumps(winsor_thresholds, indent=2),
    winsor_counts=json.dumps(winsor_counts, indent=2),
)
(DOCS_PATH / "resumen_antes_vs_despues.md").write_text(resumen_md, encoding="utf-8")

trace_lines = [
    "# Trazabilidad de Datos\n",
    "\n",
    f"- Filas iniciales: {initial_rows}",
    f"- Filas finales: {final_rows}",
    "\n",
    "## Porcentaje de filas afectadas por regla",
]

def pct(x):
    return 0 if initial_rows == 0 else round(100.0 * x / initial_rows, 4)

trace_lines.extend([
    f"- drop_leakage_cols: {pct(rule_stats['drop_leakage_cols']['rows_affected'])}%",
    f"- drop_redundant_cols: {pct(rule_stats['drop_redundant_cols']['rows_affected'])}%",
    f"- fix_sched_minute_of_day: {pct(rule_stats['fix_sched_minute_of_day']['rows_affected'])}%",
    f"- fix_precip_negative: {pct(rule_stats['fix_precip_negative']['rows_affected'])}%",
    f"- winsorize_outliers: {pct(rule_stats['winsorize_outliers']['rows_affected'])}%",
])

(DOCS_PATH / "trazabilidad_datos.md").write_text("\n".join(trace_lines), encoding="utf-8")

print("Done")
