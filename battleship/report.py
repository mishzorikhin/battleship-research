"""Markdown-отчёт по сохранённому турниру: рейтинги и парные сравнения."""
import itertools

from .strategies import ATTACKERS
from .tournament import load, paired_diff, summarize


def _title(name):
    return ATTACKERS[name].title if name in ATTACKERS else name


def build(path):
    payload = load(path)
    res = payload["results"]
    placements = list(payload["placements"])
    attackers = list(res)
    meta = payload["meta"]
    lines = [
        "# Результаты турнира (автоматически)",
        "",
        f"Партий на пару (тактика × расстановка): **{meta['games']}**, seed от {meta['offset']}. "
        f"Поле 10×10, флот {meta['fleet']}. Сравнения парные: все тактики играют одни и те же флоты.",
        "",
    ]
    for p in placements:
        rows = sorted(((summarize(res[a][p]), a) for a in attackers), key=lambda x: x[0]["mean"])
        lines += [
            f"## Расстановка защитника: {payload['placements'][p]}",
            "",
            "| # | Тактика | Среднее | ±95% | Медиана | p10 | p90 | Макс |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for rank, (s, a) in enumerate(rows, 1):
            lines.append(
                f"| {rank} | {_title(a)} | **{s['mean']:.2f}** | ±{s['ci95']:.2f} | "
                f"{s['median']:.0f} | {s['p10']:.0f} | {s['p90']:.0f} | {s['max']:.0f} |"
            )
        lines.append("")

    lines += ["## Средние значения по всем расстановкам", "", "| Тактика | " + " | ".join(payload["placements"][p] for p in placements) + " | Среднее |", "|---|" + "---:|" * (len(placements) + 1)]
    for a in attackers:
        means = [summarize(res[a][p])["mean"] for p in placements]
        lines.append(f"| {_title(a)} | " + " | ".join(f"{m:.2f}" for m in means) + f" | **{sum(means)/len(means):.2f}** |")
    lines.append("")

    lines += ["## Парные сравнения на стандартной расстановке (random)", ""]
    base = "random" if "random" in placements else placements[0]
    lines += [
        "Строка — разница (строка − столбец) средних выстрелов; знак минус значит, что тактика из строки быстрее.",
        "",
    ]
    order = [a for _, a in sorted(((summarize(res[a][base])["mean"], a) for a in attackers))]
    header = "| | " + " | ".join(_title(a) for a in order) + " |"
    lines += [header, "|---|" + "---:|" * len(order)]
    for a in order:
        cells = []
        for b in order:
            if a == b:
                cells.append("—")
                continue
            d = paired_diff(res[a][base], res[b][base])
            cells.append(f"{d['mean']:+.1f}")
        lines.append(f"| {_title(a)} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Доля партий, где первая тактика быстрее второй (random)", ""]
    for a, b in itertools.combinations(order[:4], 2):
        d = paired_diff(res[a][base], res[b][base])
        lines.append(f"- {_title(a)} vs {_title(b)}: {100*d['a_wins']:.1f}% партий за первую, разница {d['mean']:+.2f} ± {d['ci95']:.2f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys
    sys.stdout.write(build(sys.argv[1]))
