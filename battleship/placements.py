"""Стратегии расстановки флота защитником.

Расстановка строится жадно: корабли ставятся от большего к меньшему, каждый
выбирается из всех допустимых положений с весом, зависящим от стратегии.
Вес > 0 всегда, поэтому любая стратегия может выставить любой флот.
"""
import random

from .engine import FLEET, PLACEMENTS, SIZE, N_CELLS, Layout, rc, inside

CENTER = (SIZE - 1) / 2
BORDER = frozenset(i for i in range(N_CELLS) if rc(i)[0] in (0, SIZE - 1) or rc(i)[1] in (0, SIZE - 1))
NEIGHBOURS8 = tuple(
    tuple(
        (r + dr) * SIZE + (c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and inside(r + dr, c + dc)
    )
    for r, c in (rc(i) for i in range(N_CELLS))
)


def _border_count(cells):
    return sum(1 for i in cells if i in BORDER)


def _mean_center_distance(cells):
    total = 0.0
    for i in cells:
        r, c = rc(i)
        total += ((r - CENTER) ** 2 + (c - CENTER) ** 2) ** 0.5
    return total / len(cells)


def _adjacent_occupied(cells, occupied):
    touched = set()
    for i in cells:
        for j in NEIGHBOURS8[i]:
            if j in occupied and j not in cells:
                touched.add(j)
    return len(touched)


def _min_chebyshev(cells, occupied):
    if not occupied:
        return SIZE
    best = SIZE
    for i in cells:
        r, c = rc(i)
        for j in occupied:
            jr, jc = rc(j)
            best = min(best, max(abs(r - jr), abs(c - jc)))
    return best


LATTICE_CELLS = frozenset(i for i in range(N_CELLS) if (rc(i)[0] + 2 * rc(i)[1]) % 5 == 0)

# Каждая функция получает (положение, занятые клетки) и возвращает вес > 0.
WEIGHTS = {
    "random": lambda cells, occ: 1.0,
    # Корабли жмутся к краю: каждая клетка на границе множит вес на 4.
    "edge": lambda cells, occ: 4.0 ** _border_count(cells),
    # Корабли тянутся к центру.
    "center": lambda cells, occ: 1.0 / (1.0 + _mean_center_distance(cells)) ** 6,
    # Корабли стоят вплотную друг к другу.
    "packed": lambda cells, occ: 1.0 + 8.0 * _adjacent_occupied(cells, occ),
    # Корабли максимально разнесены по полю.
    "spread": lambda cells, occ: float(_min_chebyshev(cells, occ)) ** 3,
    # Защитник, знающий решётку атакующего: каждая клетка решётки снижает вес в 5 раз.
    "anti_lattice": lambda cells, occ: 0.2 ** sum(1 for i in cells if i in LATTICE_CELLS),
}

PLACEMENT_NAMES = {
    "random": "Равномерно случайная",
    "edge": "У края",
    "center": "Ближе к центру",
    "packed": "Плотно, вплотную",
    "spread": "Максимально разнесённая",
    "anti_lattice": "Против решётки",
}


def random_layout(rng, strategy="random"):
    weight = WEIGHTS[strategy]
    occupied = set()
    ships = []
    for length in sorted(FLEET, reverse=True):
        candidates = [
            cells
            for cells in PLACEMENTS[length]
            if not any(i in occupied for i in cells)
        ]
        weights = [weight(cells, occupied) for cells in candidates]
        cells = rng.choices(candidates, weights=weights, k=1)[0]
        ships.append(cells)
        occupied.update(cells)
    layout = Layout(ships)
    assert layout.is_valid()
    return layout


def fleet_rng(strategy, seed):
    """Детерминированный генератор флота: одна и та же пара (стратегия, seed)
    даёт одну и ту же расстановку для всех атакующих — сравнение парное."""
    return random.Random(f"layout|{strategy}|{seed}")
