"""Тактики атаки.

Каждый атакующий — функция choose(grid, remaining, rng) -> индекс клетки.
Он видит только свою доску (grid) и размеры ещё не потопленных кораблей.

Конструкция «простой» тактики = idle-политика (что стрелять, пока нет
подбитых кораблей) + finisher (добивание подбитых). Так можно сравнивать
паттерны стрельбы отдельно от способа добивания.
"""
from dataclasses import dataclass
from typing import Callable

from .engine import MISS, HIT, SUNK, UNKNOWN, N_CELLS, SIZE, PLACEMENTS, rc, inside

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CENTER = (SIZE - 1) / 2


# ---------- общие помощники ----------

def _first_unknown(grid, order):
    for i in order:
        if grid[i] == UNKNOWN:
            return i
    return None


def _random_unknown(grid, rng):
    cands = [i for i in range(N_CELLS) if grid[i] == UNKNOWN]
    return rng.choice(cands) if cands else None


def _has_hits(grid):
    return HIT in grid


# ---------- порядки стрельбы (idle) ----------

def _center_distance2(i):
    r, c = rc(i)
    return (r - CENTER) ** 2 + (c - CENTER) ** 2


ORDER_SEQUENTIAL = tuple(range(N_CELLS))
ORDER_PARITY = tuple(
    [i for i in range(N_CELLS) if sum(rc(i)) % 2 == 0]
    + [i for i in range(N_CELLS) if sum(rc(i)) % 2 == 1]
)
ORDER_CENTER = tuple(sorted(range(N_CELLS), key=lambda i: (_center_distance2(i), i)))
ORDER_EDGE = tuple(sorted(
    range(N_CELLS),
    key=lambda i: (not (rc(i)[0] in (0, SIZE - 1) or rc(i)[1] in (0, SIZE - 1)), _center_distance2(i), i),
))
# Решётка (r + 2c) mod 5 == 0: ровно 20 клеток, и любые 5 подряд идущих клеток
# строки или столбца содержат хотя бы одну из них. Значит, 5-клеточный корабль
# гарантированно будет задет, не зная расположения.
_LATTICE = [i for i in range(N_CELLS) if (rc(i)[0] + 2 * rc(i)[1]) % 5 == 0]
ORDER_LATTICE = tuple(_LATTICE + [i for i in range(N_CELLS) if i not in set(_LATTICE)])


def _idle_random(grid, remaining, rng):
    return _random_unknown(grid, rng)


def _idle_order(order):
    def idle(grid, remaining, rng):
        return _first_unknown(grid, order)
    return idle


# ---------- добивание (finisher) ----------

def hunt(grid, remaining, rng):
    """Классическая охота: стрелять рядом с подбитыми клетками.

    Если две подбитые клетки стоят на одной линии, приоритет у продолжения
    этой линии — так корабль добивается быстрее, чем перебором соседей.
    """
    best, best_pri = [], 0
    for i in range(N_CELLS):
        if grid[i] != HIT:
            continue
        r, c = rc(i)
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not inside(nr, nc) or grid[nr * SIZE + nc] != UNKNOWN:
                continue
            pr, pc = r - dr, c - dc
            in_line = inside(pr, pc) and grid[pr * SIZE + pc] == HIT
            pri = 2 if in_line else 1
            j = nr * SIZE + nc
            if pri > best_pri:
                best, best_pri = [j], pri
            elif pri == best_pri:
                best.append(j)
    return rng.choice(best) if best else None


def sonar(grid, remaining, rng):
    """Нелогичный добив: после попадания стрелять через клетку, а не в соседнюю.

    Соседняя клетка может быть пустой; выстрел на расстояние 2 проверяет сразу
    два пространства — если он попал, то промежуточная клетка тоже кандидат.
    Если такого выстрела нет, падаем обратно в охоту.
    """
    cands = []
    for i in range(N_CELLS):
        if grid[i] != HIT:
            continue
        r, c = rc(i)
        for dr, dc in DIRS:
            mr, mc = r + dr, c + dc
            fr, fc = r + 2 * dr, c + 2 * dc
            if inside(fr, fc) and grid[mr * SIZE + mc] == UNKNOWN and grid[fr * SIZE + fc] == UNKNOWN:
                cands.append(fr * SIZE + fc)
    if cands:
        return rng.choice(cands)
    return hunt(grid, remaining, rng)


# ---------- плотность вероятностей (PD) ----------

def pd_scores(grid, remaining, hit_boost, prior=None):
    """Сколько допустимых расположений оставшихся кораблей покрывает каждую клетку.

    Расположение недопустимо, если накрывает промах или потопленный корабль.
    hit_boost — множитель веса расположения за каждую подбитую клетку, которую
    оно накрывает: подбитая клетка почти наверняка принадлежит кораблю.
    prior — необязательный априорный вес клетки (предположение о стиле защитника).
    """
    score = [0.0] * N_CELLS
    for length in remaining:
        for cells in PLACEMENTS[length]:
            ok = True
            hits = 0
            free = []
            for i in cells:
                g = grid[i]
                if g == MISS or g == SUNK:
                    ok = False
                    break
                if g == UNKNOWN:
                    free.append(i)
                elif g == HIT:
                    hits += 1
            if not ok:
                continue
            w = 1.0 + hit_boost * hits
            if prior is not None:
                w *= sum(prior[i] for i in cells) / len(cells)
            for i in free:
                score[i] += w
    return score


def _argmax_unknown(grid, score, rng, restrict=None):
    best_val = -1.0
    best = []
    for i in range(N_CELLS):
        if grid[i] != UNKNOWN:
            continue
        if restrict is not None and not restrict(i):
            continue
        v = score[i]
        if v > best_val:
            best_val, best = v, [i]
        elif v == best_val:
            best.append(i)
    return rng.choice(best) if best else None


def _argmin_unknown(grid, score, rng):
    best_val = float("inf")
    best = []
    for i in range(N_CELLS):
        if grid[i] != UNKNOWN:
            continue
        v = score[i]
        if v < best_val:
            best_val, best = v, [i]
        elif v == best_val:
            best.append(i)
    return rng.choice(best) if best else None


HIT_BOOST = 10.0


def pd_hint(grid, remaining, rng):
    return _argmax_unknown(grid, pd_scores(grid, remaining, HIT_BOOST), rng)


def pd_plain(grid, remaining, rng):
    return _argmax_unknown(grid, pd_scores(grid, remaining, 0.0), rng)


# Априорные веса клеток: что атакующий «думает» о защитнике.
EDGE_PRIOR = tuple(4.0 if (rc(i)[0] in (0, SIZE - 1) or rc(i)[1] in (0, SIZE - 1)) else 1.0 for i in range(N_CELLS))
CENTER_PRIOR = tuple(max(0.2, 6.0 - ((rc(i)[0] - CENTER) ** 2 + (rc(i)[1] - CENTER) ** 2) ** 0.5) for i in range(N_CELLS))


def pd_edge_prior(grid, remaining, rng):
    return _argmax_unknown(grid, pd_scores(grid, remaining, HIT_BOOST, prior=EDGE_PRIOR), rng)


def pd_center_prior(grid, remaining, rng):
    return _argmax_unknown(grid, pd_scores(grid, remaining, HIT_BOOST, prior=CENTER_PRIOR), rng)


def pd_parity(grid, remaining, rng):
    """PD с шахматным ограничением, пока нет подбитых клеток."""
    if _has_hits(grid):
        return pd_hint(grid, remaining, rng)
    score = pd_scores(grid, remaining, 0.0)
    cell = _argmax_unknown(grid, score, rng, restrict=lambda i: sum(rc(i)) % 2 == 0)
    return cell if cell is not None else _argmax_unknown(grid, score, rng)


def lattice_pd(grid, remaining, rng):
    """Решётка, пока нет подбитых клеток; затем PD с бонусом за попадания."""
    if _has_hits(grid):
        return pd_hint(grid, remaining, rng)
    return _first_unknown(grid, ORDER_LATTICE)


def anti_pd(grid, remaining, rng):
    """Контроль: стрелять в клетку, которую меньше всего может занимать флот."""
    return _argmin_unknown(grid, pd_scores(grid, remaining, 0.0), rng)


# ---------- детерминизация Монте-Карло ----------

MC_SAMPLES = 80
MC_MAX_ATTEMPTS = 1500


def _sample_fleets(grid, remaining, rng):
    """Случайные расстановки оставшихся кораблей, совместимые с наблюдением.

    Совместимость: не накрывают промахи и потопленные корабли, и накрывают
    все подбитые клетки (они принадлежат ещё не потопленным кораблям).
    """
    hits = [i for i in range(N_CELLS) if grid[i] == HIT]
    lengths = sorted(remaining, reverse=True)
    valid = {
        length: [
            cells
            for cells in PLACEMENTS[length]
            if all(grid[i] != MISS and grid[i] != SUNK for i in cells)
        ]
        for length in set(lengths)
    }
    if any(not valid[length] for length in lengths):
        return []

    samples = []
    attempts = 0
    while len(samples) < MC_SAMPLES and attempts < MC_MAX_ATTEMPTS:
        attempts += 1
        occupied = set()
        ok = True
        for length in lengths:
            placed = False
            for _ in range(40):
                cells = rng.choice(valid[length])
                if not any(i in occupied for i in cells):
                    occupied.update(cells)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and all(h in occupied for h in hits):
            samples.append(occupied)
    return samples


def mc_det(grid, remaining, rng):
    """Монте-Карло: выбрать клетку, которая чаще всего занята в совместимых флотах.

    В отличие от PD учитывает, что корабли не могут перекрываться, и что каждая
    подбитая клетка принадлежит какому-то из кораблей.
    """
    samples = _sample_fleets(grid, remaining, rng)
    if not samples:
        return pd_hint(grid, remaining, rng)
    counts = [0] * N_CELLS
    for occupied in samples:
        for i in occupied:
            counts[i] += 1
    return _argmax_unknown(grid, counts, rng)


# ---------- сборка тактик ----------

@dataclass(frozen=True)
class Attacker:
    name: str
    title: str
    description: str
    choose: Callable


def _compose(finisher, idle):
    """Сначала добивание подбитых (finisher), иначе — порядок поиска (idle)."""
    def choose(grid, remaining, rng):
        cell = finisher(grid, remaining, rng)
        if cell is None:
            cell = idle(grid, remaining, rng)
        return cell
    return choose


def _pattern_hunt(idle):
    return _compose(hunt, idle)


ATTACKERS = {}


def _register(name, title, description, choose):
    ATTACKERS[name] = Attacker(name, title, description, choose)


_register("random_hunt", "Случайно + охота",
          "Случайный выстрел, после попадания — охота. Нижняя граница.",
          _pattern_hunt(_idle_random))
_register("sequential_hunt", "Построчно + охота",
          "Сканирование строк слева направо, после попадания — охота.",
          _pattern_hunt(_idle_order(ORDER_SEQUENTIAL)))
_register("parity_hunt", "Шахматка + охота",
          "Классика: сначала клетки одного цвета (любой корабль ≥2 их задевает), затем охота.",
          _pattern_hunt(_idle_order(ORDER_PARITY)))
_register("center_hunt", "От центра + охота",
          "Спираль от центра: центральные клетки покрываются большим числом положений.",
          _pattern_hunt(_idle_order(ORDER_CENTER)))
_register("edge_hunt", "От края + охота",
          "Сначала периметр, потом внутренние клетки — обратная логика центра.",
          _pattern_hunt(_idle_order(ORDER_EDGE)))
_register("lattice_hunt", "Решётка (5) + охота",
          "Точечный паттерн из 20 клеток, гарантированно задевающий любой 5-клеточный корабль.",
          _pattern_hunt(_idle_order(ORDER_LATTICE)))
_register("pd_hint", "PD с бонусом за попадания",
          "Плотность вероятностей по допустимым расположениям флота; попадания усиливают вес.",
          pd_hint)
_register("pd_plain", "PD без бонуса",
          "Та же плотность, но попадания ничего не добавляют к весу расположения.",
          pd_plain)
_register("pd_parity", "PD + шахматка",
          "PD, пока нет подбитых клеток, — только в клетках одного цвета.",
          pd_parity)
_register("lattice_pd", "Решётка + PD",
          "Гибрид: решётка до первого попадания, потом PD с бонусом.",
          lattice_pd)
_register("mc_det", "Монте-Карло (детерминизация)",
          "Сэмплирование совместимых флотов; выбор клетки, занятой чаще всего.",
          mc_det)
_register("anti_pd", "Анти-PD (контроль)",
          "Стрельба в наименее вероятные клетки + охота. Проверка, что PD важна.",
          _pattern_hunt(lambda g, r, rng: _argmin_unknown(g, pd_scores(g, r, 0.0), rng)))

_register("sonar_hunt", "Сонар: выстрел через клетку",
          "Шахматка, а после попадания — выстрел через клетку вдоль оси (нелогичный добив).",
          _compose(sonar, _idle_order(ORDER_PARITY)))
_register("pd_edge_prior", "PD + априори «край»",
          "PD с предположением, что защитник ставит корабли у края.",
          pd_edge_prior)
_register("pd_center_prior", "PD + априори «центр»",
          "PD с предположением, что защитник ставит корабли в центре.",
          pd_center_prior)

ATTACKER_ORDER = tuple(ATTACKERS)
