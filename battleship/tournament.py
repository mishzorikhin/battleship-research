"""Прогон серий партий и статистика.

Все атакующие играют одни и те же расстановки (пара: стратегия защитника, seed),
поэтому разницу между тактиками можно считать парно — это сильно снижает шум.
"""
import json
import math
import multiprocessing as mp
import random
import statistics
from datetime import datetime, timezone

from .engine import play
from .placements import PLACEMENT_NAMES, fleet_rng, random_layout
from .strategies import ATTACKERS


def run_one(task):
    attacker, placement, seed = task
    layout = random_layout(fleet_rng(placement, seed), placement)
    rng = random.Random(f"attack|{attacker}|{placement}|{seed}")
    shots, _ = play(layout, ATTACKERS[attacker].choose, rng)
    return attacker, placement, seed, shots


def _tasks(attackers, placements, games, offset=0):
    for placement in placements:
        for attacker in attackers:
            for seed in range(offset, offset + games):
                yield (attacker, placement, seed)


def run_tournament(attackers, placements, games, workers=None, progress=None, offset=0):
    """Возвращает {attacker: {placement: [shots по seed]}}."""
    attackers = list(attackers)
    placements = list(placements)
    total = len(attackers) * len(placements) * games
    workers = workers or max(1, mp.cpu_count())
    results = {a: {p: [None] * games for p in placements} for a in attackers}
    done = 0
    # spawn, а не fork: турнир может стартовать из фонового потока сервера.
    with mp.get_context("spawn").Pool(workers) as pool:
        for attacker, placement, seed, shots in pool.imap_unordered(
            run_one, _tasks(attackers, placements, games, offset), chunksize=16
        ):
            results[attacker][placement][seed - offset] = shots
            done += 1
            if progress and (done % 50 == 0 or done == total):
                progress(done, total)
    return results


def summarize(values):
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) if n > 1 else 0.0
    ordered = sorted(values)

    def pct(p):
        k = (n - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95": 1.96 * sd / math.sqrt(n) if n > 1 else 0.0,
        "median": pct(0.5),
        "p10": pct(0.1),
        "p90": pct(0.9),
        "min": ordered[0],
        "max": ordered[-1],
    }


def paired_diff(a_vals, b_vals):
    """Парная разница (a - b): среднее, 95% ДИ и доля партий, где a быстрее."""
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.pstdev(diffs) if n > 1 else 0.0
    wins = sum(1 for d in diffs if d < 0) / n
    return {"mean": mean, "ci95": 1.96 * sd / math.sqrt(n) if n > 1 else 0.0, "a_wins": wins}


def save(results, path, games, offset=0):
    payload = {
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "games": games,
            "offset": offset,
            "fleet": [5, 4, 3, 3, 2],
            "board": 10,
        },
        "attackers": {
            a: {"title": ATTACKERS[a].title, "description": ATTACKERS[a].description}
            for a in results
        },
        "placements": {p: PLACEMENT_NAMES[p] for p in next(iter(results.values()))},
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return payload


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


