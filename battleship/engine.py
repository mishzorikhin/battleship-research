"""Ядро морского боя: поле, флот, расположения кораблей и ход партии.

Клетки поля адресуются плоским индексом i = row * SIZE + col.
Корабли не могут перекрываться и не могут касаться друг друга ни сторонами,
ни углами (русские правила). Из этого следуют выводы, которые движок
делает за атакующего (он «ведёт учёт» идеально):
  * диагональные соседи подбитой клетки пусты — корабль прямой, а чужой
    корабль не может касаться углом;
  * после потопления все клетки вокруг корабля пусты.
Такие клетки в доске атакующего помечаются MISS (и не считаются выстрелами).
"""
import random

SIZE = 10
N_CELLS = SIZE * SIZE
FLEET = (5, 4, 3, 3, 2)
TOTAL_SHIP_CELLS = sum(FLEET)

# Состояние клетки с точки зрения атакующего.
UNKNOWN, MISS, HIT, SUNK = -1, 0, 1, 2


def idx(row, col):
    return row * SIZE + col


def rc(i):
    return divmod(i, SIZE)


def inside(row, col):
    return 0 <= row < SIZE and 0 <= col < SIZE


def _neighbours(i, diagonal_only=False):
    r, c = rc(i)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not (dr or dc) or not inside(r + dr, c + dc):
                continue
            if diagonal_only and not (dr and dc):
                continue
            out.append(idx(r + dr, c + dc))
    return tuple(out)


NEIGHBOURS8 = tuple(_neighbours(i) for i in range(N_CELLS))
DIAGONALS = tuple(_neighbours(i, diagonal_only=True) for i in range(N_CELLS))


def _build_placements():
    """Все возможные положения корабля каждой длины (горизонтальные и вертикальные)."""
    table = {}
    for length in set(FLEET):
        out = []
        for row in range(SIZE):
            for col in range(SIZE - length + 1):
                out.append(tuple(idx(row, col + k) for k in range(length)))
        for col in range(SIZE):
            for row in range(SIZE - length + 1):
                out.append(tuple(idx(row + k, col) for k in range(length)))
        table[length] = tuple(out)
    return table


PLACEMENTS = _build_placements()


class Layout:
    """Реальная расстановка флота защитника."""

    __slots__ = ("ships", "owner")

    def __init__(self, ships):
        self.ships = tuple(tuple(ship) for ship in ships)
        self.owner = [-1] * N_CELLS
        for k, ship in enumerate(self.ships):
            for i in ship:
                self.owner[i] = k

    def is_valid(self):
        """Флот того же состава, корабли не пересекаются и не касаются."""
        lengths = sorted(len(s) for s in self.ships)
        if lengths != sorted(FLEET):
            return False
        cells = [i for s in self.ships for i in s]
        if not len(cells) == len(set(cells)) == TOTAL_SHIP_CELLS:
            return False
        for k, ship in enumerate(self.ships):
            for i in ship:
                for j in NEIGHBOURS8[i]:
                    if self.owner[j] not in (-1, k):
                        return False
        return True

    def to_dict(self):
        return [[list(rc(i)) for i in ship] for ship in self.ships]


def _mark_empty(grid, cells):
    """Пометить неизвестные клетки как пустые; вернуть список помеченных."""
    marked = []
    for j in cells:
        if grid[j] == UNKNOWN:
            grid[j] = MISS
            marked.append(j)
    return marked


def play(layout, choose, rng, record=False, deduce=True):
    """Играет одну партию до полного потопления флота.

    choose(grid, remaining, rng) -> индекс неизвестной клетки.
    grid      — список из N_CELLS состояний (UNKNOWN/MISS/HIT/SUNK);
    remaining — длины кораблей, которые ещё не потоплены (их атакующий видит
                в сообщениях «потоплен», поэтому знать размеры законно).
    deduce — автоматически помечать клетки, пустота которых следует из правил
    (диагонали подбитой клетки, окружение потопленного корабля).
    Возвращает (число выстрелов, журнал ходов или None); запись журнала —
    (клетка, исход, список выведенных пустых клеток).
    """
    grid = [UNKNOWN] * N_CELLS
    hits_on_ship = [0] * len(layout.ships)
    sunk = [False] * len(layout.ships)
    cells_left = TOTAL_SHIP_CELLS
    shots = 0
    log = [] if record else None

    while cells_left:
        remaining = [len(s) for k, s in enumerate(layout.ships) if not sunk[k]]
        i = choose(grid, remaining, rng)
        if grid[i] != UNKNOWN:
            raise RuntimeError(f"атакующий повторил выстрел в клетку {rc(i)}")
        shots += 1
        k = layout.owner[i]
        inferred = []
        if k < 0:
            grid[i] = MISS
            outcome = "miss"
        else:
            cells_left -= 1
            hits_on_ship[k] += 1
            ship = layout.ships[k]
            if hits_on_ship[k] == len(ship):
                sunk[k] = True
                for j in ship:
                    grid[j] = SUNK
                outcome = "sunk"
                if deduce:
                    inferred = _mark_empty(grid, [n for j in ship for n in NEIGHBOURS8[j]])
            else:
                grid[i] = HIT
                outcome = "hit"
                if deduce:
                    inferred = _mark_empty(grid, DIAGONALS[i])
        if record:
            log.append((i, outcome, inferred))
    return shots, log
