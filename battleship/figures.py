"""Картинки для исследования: python -m battleship.figures [папка]

Тепловые карты расстановок и стрельбы, карта вероятностей PD, матрица
результатов, рейтинг, распределения, найденные флоты. Требуется matplotlib.
"""
import json
import os
import random
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .engine import FLEET, HIT, MISS, SIZE, SUNK, UNKNOWN, Layout, play, rc
from .placements import PLACEMENT_NAMES, WEIGHTS, random_layout
from .strategies import ATTACKERS, pd_scores
from .tournament import load, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
HIT_RED, SUNK_DARK = "#e34948", "#7c2d12"
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#a9c9f2", BLUE, "#123f80"])
COLS = "абвгдежзик"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
})


def _ink_on(value):
    return "#ffffff" if value > 0.55 else INK


def heat(ax, matrix, vmin, vmax, title=None, annotate=False, fmt="{:.0f}", ticks=True, aspect="equal"):
    ax.pcolormesh(matrix, cmap=SEQ, vmin=vmin, vmax=vmax, edgecolors=SURFACE, linewidth=1.5)
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(matrix.shape[0], 0)
    ax.set_aspect(aspect)
    for s in ax.spines.values():
        s.set_visible(False)
    if ticks and matrix.shape == (SIZE, SIZE):
        ax.set_xticks(np.arange(SIZE) + 0.5, list(COLS), fontsize=8)
        ax.set_yticks(np.arange(SIZE) + 0.5, [str(i + 1) for i in range(SIZE)], fontsize=8)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    if annotate:
        for (r, c), v in np.ndenumerate(matrix):
            ax.text(c + 0.5, r + 0.5, fmt.format(v), ha="center", va="center", fontsize=8,
                    color=_ink_on((v - vmin) / (vmax - vmin + 1e-9)))


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("сохранено:", path)
    return path


def occupancy(strategy, n=2000):
    grid = np.zeros((SIZE, SIZE))
    for seed in range(n):
        layout = random_layout(random.Random(f"fig|{strategy}|{seed}"), strategy)
        for ship in layout.ships:
            for i in ship:
                grid[rc(i)] += 1
    return 100 * grid / n


def fig_placements(out):
    names = list(PLACEMENT_NAMES)
    maps = {p: occupancy(p) for p in names}
    vmax = max(m.max() for m in maps.values())
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.2))
    for ax, p in zip(axes.flat, names):
        heat(ax, maps[p], 0, vmax, PLACEMENT_NAMES[p], annotate=True)
    fig.suptitle("Где защитник ставит корабли: вероятность, что клетка занята, %", fontsize=13, x=0.06, ha="left")
    fig.text(0.06, 0.935, "2000 расстановок на стратегию; в среднем занято 17% клеток, шкала общая", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3.5)
    return _save(fig, out, "1_placements_heatmaps.png")


def shot_frequency(attacker, n=300, first=25):
    grid = np.zeros((SIZE, SIZE))
    for seed in range(n):
        layout = random_layout(random.Random(f"shots|{seed}"))
        _, log = play(layout, ATTACKERS[attacker].choose, random.Random(f"shots|{attacker}|{seed}"), record=True)
        for i, _, _ in log[:first]:
            grid[rc(i)] += 1
    return 100 * grid / n


def fig_shot_patterns(out):
    names = ["sequential_hunt", "random_hunt", "parity_hunt", "center_hunt",
             "edge_hunt", "lattice_hunt", "pd_hint", "pd_edge_prior"]
    maps = {a: shot_frequency(a) for a in names}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.6))
    for ax, a in zip(axes.flat, names):
        heat(ax, maps[a], 0, 100, ATTACKERS[a].title)
    fig.suptitle("Куда стреляют тактики: доля партий, где клетка обстреляна в первых 25 выстрелах, %", fontsize=13, x=0.04, ha="left")
    fig.text(0.04, 0.925, "300 партий против равномерной расстановки; клетки, закрытые выводом по правилам, выстрелами не считаются", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.5)
    return _save(fig, out, "2_shot_patterns.png")


def _record_states(attacker, seed):
    layout = random_layout(random.Random(f"pdmap|{seed}"))
    states = []
    base = ATTACKERS[attacker].choose

    def choose(grid, remaining, rng):
        states.append((list(grid), list(remaining)))
        return base(grid, remaining, rng)

    play(layout, choose, random.Random(f"pdmap|a|{seed}"))
    return states


def fig_pd_map(out, seed=11):
    states = _record_states("pd_hint", seed)
    first_hit = next(k for k, (g, _) in enumerate(states) if HIT in g)
    steps = [0, max(first_hit - 1, 1), first_hit + 1, first_hit + 4, min(len(states) - 1, first_hit + 14)]
    fig, axes = plt.subplots(1, len(steps), figsize=(3.3 * len(steps), 4.4))
    for ax, k in zip(axes, steps):
        grid, remaining = states[k]
        score = np.array(pd_scores(grid, remaining, 10.0)).reshape(SIZE, SIZE)
        known = np.array(grid).reshape(SIZE, SIZE) != UNKNOWN
        prob = np.where(known, 0, 100 * score / score.sum())
        heat(ax, prob, 0, max(prob.max(), 1), f"после {k} выстрелов")
        for (r, c), g in np.ndenumerate(np.array(grid).reshape(SIZE, SIZE)):
            if g == MISS:
                ax.plot(c + 0.5, r + 0.5, "x", color=INK2, ms=5, mew=1.2)
            elif g in (HIT, SUNK):
                ax.add_patch(plt.Circle((c + 0.5, r + 0.5), 0.34, color=HIT_RED if g == HIT else SUNK_DARK))
    fig.suptitle("Карта вероятностей PD: где тактика ждёт корабль", fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.905, "цвет — доля веса клетки среди всех допустимых положений оставшихся кораблей; × промах, красный — попадание, тёмный — потоплено",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, out, "3_pd_probability_map.png")


def fig_matrix(out, data):
    R, placements = data["results"], list(data["placements"])
    avg = {a: np.mean([summarize(R[a][p])["mean"] for p in placements]) for a in R}
    order = sorted(R, key=avg.get)
    M = np.array([[summarize(R[a][p])["mean"] for p in placements] for a in order])
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    heat(ax, M, 28, 85, annotate=True, fmt="{:.1f}", ticks=False, aspect="auto")
    ax.set_yticks(np.arange(len(order)) + 0.5, [ATTACKERS[a].title for a in order], fontsize=9.5)
    ax.set_xticks(np.arange(len(placements)) + 0.5, ["\n".join(textwrap.wrap(data["placements"][p], 14)) for p in placements], fontsize=9.5)
    ax.xaxis.tick_top()
    for j in range(M.shape[1]):   # лучший в колонке — рамка
        i = int(M[:, j].argmin())
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, ec=ORANGE, lw=2.5))
    fig.text(0.01, 0.99, "Среднее число выстрелов: тактика × расстановка защитника (темнее — хуже)", fontsize=13, va="top")
    fig.text(0.01, 0.955, "оранжевая рамка — лучшая тактика против этой расстановки; строки отсортированы по среднему", fontsize=9.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, out, "4_matrix_attacker_x_placement.png")


def fig_ranking(out, data, placement="random"):
    R = data["results"]
    stats = {a: summarize(R[a][placement]) for a in R}
    order = sorted(R, key=lambda a: stats[a]["mean"])
    fig, ax = plt.subplots(figsize=(10, 6.6))
    y = np.arange(len(order))
    ax.barh(y, [stats[a]["mean"] for a in order], xerr=[stats[a]["ci95"] for a in order],
            color=BLUE, height=0.62, error_kw={"ecolor": INK2, "elinewidth": 1.2, "capsize": 3})
    for k, a in enumerate(order):
        ax.text(stats[a]["mean"] + stats[a]["ci95"] + 0.8, k, f"{stats[a]['mean']:.1f}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [ATTACKERS[a].title for a in order], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 75)
    ax.set_xlabel("выстрелов до победы, среднее и 95% доверительный интервал (1000 партий)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Рейтинг тактик на равномерной расстановке (меньше — лучше)", fontsize=13, loc="left", pad=12)
    fig.tight_layout()
    return _save(fig, out, "5_ranking.png")


def fig_distribution(out, data, placement="random"):
    R = data["results"]
    picks = [("pd_hint", BLUE), ("parity_hunt", ORANGE), ("random_hunt", AQUA)]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bins = np.arange(18, 100, 2)
    for a, color in picks:
        v = np.array(R[a][placement])
        ax.hist(v, bins=bins, histtype="stepfilled", alpha=0.28, color=color)
        ax.hist(v, bins=bins, histtype="step", color=color, lw=2, label=f"{ATTACKERS[a].title}: среднее {v.mean():.1f}")
        ax.axvline(v.mean(), color=color, lw=1.2, ls="--")
    ax.set_xlabel("выстрелов до победы")
    ax.set_ylabel("партий из 1000")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.set_title("Разброс результата: PD быстрее и стабильнее случайной охоты", fontsize=13, loc="left", pad=12)
    fig.tight_layout()
    return _save(fig, out, "6_distributions.png")


def _draw_layout(ax, ships, title):
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)
    ax.set_aspect("equal")
    ax.pcolormesh(np.zeros((SIZE, SIZE)), cmap=LinearSegmentedColormap.from_list("e", ["#f1f0ec", "#f1f0ec"]),
                  edgecolors=SURFACE, linewidth=1.5)
    shades = ["#123f80", "#1f5fb0", BLUE, "#5b9be4", "#8fbbee"]
    for k, ship in enumerate(ships):
        for i in ship:
            r, c = divmod(i, SIZE)
            ax.add_patch(plt.Rectangle((c + 0.04, r + 0.04), 0.92, 0.92, color=shades[k]))
        r, c = divmod(ship[len(ship) // 2], SIZE)
        ax.text(c + 0.5, r + 0.5, str(len(ship)), color="#fff", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(SIZE) + 0.5, list(COLS), fontsize=8)
    ax.set_yticks(np.arange(SIZE) + 0.5, [str(i + 1) for i in range(SIZE)], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, loc="left", pad=8)


def fig_layouts(out):
    panels = []
    for label, fname in (("Лучший ответ на PD", "best_layout_pd_hint.json"), ("Maximin-флот", "best_layout_robust.json")):
        path = os.path.join(ROOT, "results", fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path, encoding="utf-8"))
        ships = [[r * SIZE + c for r, c in s] for s in d["ships"]]
        assert Layout(ships).is_valid()
        if "validation" in d:
            v = d["validation"]
            worst = min(v, key=lambda a: v[a]["mean"])
            sub = f"против PD {v['pd_hint']['mean']:.1f}, худшая тактика {ATTACKERS[worst].title}: {v[worst]['mean']:.1f}"
        else:
            sub = ""
        panels.append((label + ("\n" + sub if sub else ""), ships))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 6.2))
    axes = np.atleast_1d(axes)
    for ax, (title, ships) in zip(axes, panels):
        _draw_layout(ax, ships, title)
    fig.tight_layout()
    return _save(fig, out, "7_best_layouts.png")


def main(out=None):
    out = out or os.path.join(ROOT, "results", "figures")
    data = load(os.path.join(ROOT, "results", "tournament.json"))
    fig_placements(out)
    fig_shot_patterns(out)
    fig_pd_map(out)
    fig_matrix(out, data)
    fig_ranking(out, data)
    fig_distribution(out, data)
    fig_layouts(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
