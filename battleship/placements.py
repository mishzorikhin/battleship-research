"""Стратегии расстановки флота защитником.

Расстановка строится жадно: корабли ставятся от большего к меньшему, каждый
выбирается из всех допустимых положений (не пересекающихся с уже стоящими
кораблями и не касающихся их даже углом) с весом, зависящим от стратегии.
Если жадный выбор зашёл в тупик, расстановка строится заново.
"""
import random

from .engine import FLEET, NEIGHBOURS8, PLACEMENTS, SIZE, N_CELLS, Layout, rc

CENTER = (SIZE - 1) / 2
BORDER = frozenset(i for i in range(N_CELLS) if rc(i)[0] in (0, SIZE - 1) or rc(i)[1] in (0, SIZE - 1))


def _border_count(cells):
    return sum(1 for i in cells if i in BORDER)


def _mean_center_distance(cells):
    total = 0.0
    for i in cells:
        r, c = rc(i)
        total += ((r - CENTER) ** 2 + (c - CENTER) ** 2) ** 0.5
    return total / len(cells)


def _near_occupied(cells, occupied):
    """Сколько занятых клеток в пределах двух клеток от положения (касаться нельзя,
    поэтому это «впритык с зазором в одну клетку»)."""
    near = set()
    for i in cells:
        r, c = rc(i)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE:
                    j = (r + dr) * SIZE + c + dc
                    if j in occupied:
                        near.add(j)
    return len(near)


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
    # Корабли стоят как можно теснее: через одну клетку друг от друга.
    "packed": lambda cells, occ: 1.0 + 8.0 * _near_occupied(cells, occ),
    # Корабли максимально разнесены по полю.
    "spread": lambda cells, occ: float(_min_chebyshev(cells, occ)) ** 3,
    # Защитник, знающий решётку атакующего: каждая клетка решётки снижает вес в 5 раз.
    "anti_lattice": lambda cells, occ: 0.2 ** sum(1 for i in cells if i in LATTICE_CELLS),
}

PLACEMENT_NAMES = {
    "random": "Равномерно случайная",
    "edge": "У края",
    "center": "Ближе к центру",
    "packed": "Тесно (через клетку)",
    "spread": "Максимально разнесённая",
    "anti_lattice": "Против решётки",
}


def _try_layout(rng, weight):
    occupied = set()
    blocked = set()  # занятые клетки и их окружение: сюда ставить нельзя
    ships = []
    for length in sorted(FLEET, reverse=True):
        candidates = [
            cells for cells in PLACEMENTS[length]
            if not any(i in blocked for i in cells)
        ]
        if not candidates:
            return None
        weights = [weight(cells, occupied) for cells in candidates]
        cells = rng.choices(candidates, weights=weights, k=1)[0]
        ships.append(cells)
        for i in cells:
            occupied.add(i)
            blocked.add(i)
            blocked.update(NEIGHBOURS8[i])
    return Layout(ships)


def random_layout(rng, strategy="random"):
    weight = WEIGHTS[strategy]
    while True:
        layout = _try_layout(rng, weight)
        if layout is not None:
            assert layout.is_valid()
            return layout


def fleet_rng(strategy, seed):
    """Детерминированный генератор флота: одна и та же пара (стратегия, seed)
    даёт одну и ту же расстановку для всех атакующих — сравнение парное."""
    return random.Random(f"layout|{strategy}|{seed}")
