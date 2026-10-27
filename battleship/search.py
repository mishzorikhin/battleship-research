"""Поиск расстановки флота, которая максимизирует число выстрелов атакующего.

Оценка расстановки — среднее по K партиям с одним и тем же флотом и разными
seed'ами атакующего (случайные ничьи в PD). Поиск — hill-climbing: переставить
один корабль на другое допустимое место, сохранить, если среднее не падает.
Итоговая расстановка проверяется на отдельных, новых seed'ах.
"""
import multiprocessing as mp
import random
import statistics

from .engine import FLEET, PLACEMENTS, Layout, play
from .placements import random_layout
from .strategies import ATTACKERS


def evaluate(ships, attacker, seeds):
    layout = Layout(ships)
    choose = ATTACKERS[attacker].choose
    return statistics.fmean(
        play(layout, choose, random.Random(f"search|{attacker}|{s}"))[0] for s in seeds
    )


def mutate(ships, rng):
    """Переставить один корабль в другое место, не пересекаясь с остальными."""
    k = rng.randrange(len(ships))
    others = {i for j, s in enumerate(ships) if j != k for i in s}
    cands = [p for p in PLACEMENTS[len(ships[k])] if not any(i in others for i in p)]
    new = list(ships)
    new[k] = rng.choice(cands)
    return tuple(new)


def worst_case(ships, attackers, seeds):
    """Maximin для защитника: худшее (для него) среднее по набору атакующих."""
    return min(evaluate(ships, a, seeds) for a in attackers)


def hill_climb(args):
    attackers, restart, iters, k_seeds = args
    rng = random.Random(f"climb|{'+'.join(attackers)}|{restart}")
    seeds = range(k_seeds)
    score = lambda ships: worst_case(ships, attackers, seeds)
    cur = tuple(random_layout(rng).ships)
    cur_val = score(cur)
    start_val = cur_val
    for _ in range(iters):
        cand = mutate(cur, rng)
        val = score(cand)
        if val >= cur_val:
            cur, cur_val = cand, val
    return restart, start_val, cur_val, cur


def search(attackers=("pd_hint",), restarts=8, iters=150, k_seeds=150, workers=None):
    tasks = [(tuple(attackers), r, iters, k_seeds) for r in range(restarts)]
    with mp.get_context("spawn").Pool(workers or mp.cpu_count()) as pool:
        return pool.map(hill_climb, tasks)


def validate(ships, attacker, n=1000, offset=100000):
    """Новые seed'ы, которых не было в поиске."""
    vals = [play(Layout(ships), ATTACKERS[attacker].choose,
                 random.Random(f"val|{attacker}|{s}"))[0] for s in range(offset, offset + n)]
    sd = statistics.pstdev(vals)
    return statistics.fmean(vals), 1.96 * sd / n ** 0.5


def describe(ships):
    grid = [["." for _ in range(10)] for _ in range(10)]
    letters = "ABCDE"
    for k, ship in enumerate(ships):
        for i in ship:
            grid[i // 10][i % 10] = letters[k]
    return "\n".join(" ".join(row) for row in grid)


if __name__ == "__main__":
    import json
    import sys

    # python -m battleship.search pd_hint            — лучший ответ на одну тактику
    # python -m battleship.search robust a1 a2 ...   — maximin по набору тактик
    args = sys.argv[1:]
    robust = bool(args) and args[0] == "robust"
    attackers = args[1:] if robust else (args[:1] or ["pd_hint"])
    label = "robust" if robust else attackers[0]
    results = search(attackers, restarts=8, iters=120 if robust else 150, k_seeds=60 if robust else 150)
    for restart, start, best, ships in sorted(results, key=lambda r: -r[2]):
        print(f"restart {restart}: старт {start:.1f} -> поиск {best:.1f}")
    best_ships = max(results, key=lambda r: r[2])[3]
    report = {a: validate(best_ships, a, n=500) for a in ATTACKERS}
    for a, (m, ci) in report.items():
        print(f"{ATTACKERS[a].title:32s} {m:6.2f} ±{ci:4.2f}")
    print(describe(best_ships))
    with open(f"results/best_layout_{label}.json", "w", encoding="utf-8") as fh:
        json.dump({
            "search_attackers": attackers,
            "ships": [[list(divmod(i, 10)) for i in s] for s in best_ships],
            "validation": {a: {"mean": m, "ci95": ci} for a, (m, ci) in report.items()},
        }, fh, ensure_ascii=False, indent=1)
