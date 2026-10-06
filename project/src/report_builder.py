from datetime import datetime
from pathlib import Path

import optuna


def generate_report(study, config, test_metric, baseline_metric, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    direction = config["tuning"]["direction"]
    primary_metric = config["search"]["scoring"]

    trials_df = study.trials_dataframe()
    completed = trials_df[trials_df["state"] == "COMPLETE"]

    asc = direction == "minimize"
    top10 = completed.sort_values("value", ascending=asc).head(10)

    n_completed = int((trials_df["state"] == "COMPLETE").sum())
    n_failed = int((trials_df["state"] == "FAIL").sum())
    n_pruned = int((trials_df["state"] == "PRUNED").sum())

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {}

    best_curve = completed["value"].cummax() if not asc else completed["value"].cummin()

    lines = []
    lines.append(f"# Отчёт по подбору: `{config['name']}`")
    lines.append("")
    lines.append(f"- **Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Версия конфига:** {config.get('version', '—')}")
    lines.append(f"- **Задача:** {config['task']}")
    lines.append(f"- **Метрика подбора:** `{primary_metric}` ({direction})")
    lines.append(f"- **CV:** {config['tuning']['cv']['folds']} фолдов")
    lines.append(f"- **Бюджет:** n_trials={config['tuning']['budget']['n_trials']}, "
                 f"timeout={config['tuning']['budget']['timeout_seconds']}s")
    lines.append(f"- **Seed:** {config['tuning']['seed']}")
    lines.append("")

    lines.append("## Итог")
    lines.append("")
    lines.append(f"- **Победитель:** `{study.best_params.get('model', '—')}`")
    lines.append(f"- **CV-метрика победителя:** {study.best_value:.4f}")
    lines.append(f"- **Baseline (default params):** {baseline_metric:.4f}")
    lines.append(f"- **Метрика на тесте:** {test_metric:.4f}")
    lines.append("")

    lines.append("## Статистика испытаний")
    lines.append("")
    lines.append(f"- Завершено: **{n_completed}**")
    lines.append(f"- Упало: **{n_failed}**")
    lines.append(f"- Отсечено (pruned): **{n_pruned}**")
    lines.append("")

    lines.append("## Лидерборд (топ-10)")
    lines.append("")
    param_cols = [c for c in top10.columns if c.startswith("params_")]
    header = "| # | value | " + " | ".join(c.replace("params_", "") for c in param_cols) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(param_cols) + 2))
    for _, row in top10.iterrows():
        cells = [str(int(row["number"]))] + [f"{row['value']:.4f}"]
        cells += [str(row[c]) for c in param_cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    if importances:
        lines.append("## Важность гиперпараметров")
        lines.append("")
        lines.append("| Параметр | Важность |")
        lines.append("|---|---|")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
            lines.append(f"| `{name}` | {imp:.4f} |")
        lines.append("")

    lines.append("## Кривая лучшего значения")
    lines.append("")
    lines.append("```")
    for i, v in enumerate(best_curve.tolist()):
        lines.append(f"trial {i:>3}: {v:.4f}")
    lines.append("```")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт сохранён: {output_path}")