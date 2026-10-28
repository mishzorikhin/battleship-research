import random
import unittest

from battleship.engine import FLEET, MISS, N_CELLS, NEIGHBOURS8, SIZE, UNKNOWN, HIT, Layout, play, rc, idx
from battleship.placements import WEIGHTS, random_layout
from battleship.strategies import ATTACKERS, ORDER_LATTICE, _sample_fleets, hunt, pd_scores, mc_det


class LayoutTests(unittest.TestCase):
    def test_every_strategy_builds_valid_fleet(self):
        for strategy in WEIGHTS:
            for seed in range(50):
                layout = random_layout(random.Random(f"{strategy}-{seed}"), strategy)
                self.assertTrue(layout.is_valid(), strategy)

    def test_ships_are_straight_and_in_bounds(self):
        layout = random_layout(random.Random(1), "random")
        for ship in layout.ships:
            rows = {rc(i)[0] for i in ship}
            cols = {rc(i)[1] for i in ship}
            self.assertTrue(len(rows) == 1 or len(cols) == 1)
            self.assertTrue(all(0 <= r < SIZE and 0 <= c < SIZE for r, c in map(rc, ship)))


class NoTouchTests(unittest.TestCase):
    def test_ships_never_touch_even_diagonally(self):
        for strategy in WEIGHTS:
            for seed in range(100):
                layout = random_layout(random.Random(f"nt-{strategy}-{seed}"), strategy)
                for k, ship in enumerate(layout.ships):
                    for i in ship:
                        for j in NEIGHBOURS8[i]:
                            self.assertIn(layout.owner[j], (-1, k), strategy)

    def test_touching_fleet_is_rejected(self):
        touching = Layout([
            [idx(0, 0), idx(0, 1), idx(0, 2), idx(0, 3), idx(0, 4)],
            [idx(1, 0), idx(1, 1), idx(1, 2), idx(1, 3)],   # касается стороной
            [idx(3, 0), idx(3, 1), idx(3, 2)],
            [idx(5, 0), idx(5, 1), idx(5, 2)],
            [idx(7, 0), idx(7, 1)],
        ])
        self.assertFalse(touching.is_valid())

    def test_diagonals_of_hit_are_marked_empty(self):
        ships = [
            [idx(5, 5), idx(5, 6), idx(5, 7), idx(5, 8), idx(5, 9)],
            [idx(0, 0), idx(0, 1), idx(0, 2), idx(0, 3)],
            [idx(2, 0), idx(2, 1), idx(2, 2)],
            [idx(8, 0), idx(8, 1), idx(8, 2)],
            [idx(9, 5), idx(9, 6)],
        ]
        layout = Layout(ships)
        self.assertTrue(layout.is_valid())
        grids = []

        def choose(grid, remaining, rng):
            grids.append(list(grid))
            return idx(5, 5) if len(grids) == 1 else next(i for i in range(N_CELLS) if grid[i] == UNKNOWN)

        play(layout, choose, random.Random(0), record=False)
        after_first_shot = grids[1]
        for cell in (idx(4, 4), idx(4, 6), idx(6, 4), idx(6, 6)):
            self.assertEqual(after_first_shot[cell], MISS)
        self.assertEqual(after_first_shot[idx(5, 4)], UNKNOWN)   # соседи по стороне неизвестны

    def test_sunk_ship_is_surrounded_by_empty_cells(self):
        ships = [
            [idx(5, 5), idx(5, 6), idx(5, 7), idx(5, 8), idx(5, 9)],
            [idx(0, 0), idx(0, 1), idx(0, 2), idx(0, 3)],
            [idx(2, 0), idx(2, 1), idx(2, 2)],
            [idx(8, 0), idx(8, 1), idx(8, 2)],
            [idx(9, 5), idx(9, 6)],
        ]
        layout = Layout(ships)
        shots = [idx(9, 5), idx(9, 6)]
        seen = {}

        def choose(grid, remaining, rng):
            if shots:
                return shots.pop(0)
            seen.setdefault("grid", list(grid))
            return next(i for i in range(N_CELLS) if grid[i] == UNKNOWN)

        play(layout, choose, random.Random(0))
        for cell in (idx(8, 4), idx(8, 5), idx(8, 6), idx(8, 7), idx(9, 4), idx(9, 7)):
            self.assertEqual(seen["grid"][cell], MISS)

    def test_monte_carlo_samples_do_not_touch(self):
        def components(cells):
            left, count = set(cells), 0
            while left:
                count += 1
                stack = [left.pop()]
                while stack:
                    for j in NEIGHBOURS8[stack.pop()]:
                        if j in left:
                            left.remove(j)
                            stack.append(j)
            return count

        grid = [UNKNOWN] * N_CELLS
        samples = _sample_fleets(grid, list(FLEET), random.Random(4))
        self.assertTrue(samples)
        for sample in samples:
            self.assertEqual(len(sample), sum(FLEET))
            self.assertEqual(components(sample), len(FLEET))  # 5 кораблей = 5 отдельных групп


class GameTests(unittest.TestCase):
    def test_every_attacker_finishes_without_repeats(self):
        for name, attacker in ATTACKERS.items():
            for seed in range(5):
                layout = random_layout(random.Random(f"L{seed}"), "random")
                shots, log = play(layout, attacker.choose, random.Random(f"A{seed}"), record=True)
                cells = [i for i, _, _ in log]
                self.assertEqual(len(cells), len(set(cells)), name)
                self.assertLessEqual(shots, N_CELLS, name)
                self.assertEqual(sum(1 for _, o, _ in log if o != "miss"), sum(FLEET), name)

    def test_hunt_targets_neighbour_of_hit(self):
        grid = [UNKNOWN] * N_CELLS
        centre = idx(5, 5)
        grid[centre] = HIT
        choice = hunt(grid, [5], random.Random(0))
        self.assertIn(choice, {idx(4, 5), idx(6, 5), idx(5, 4), idx(5, 6)})

    def test_hunt_extends_line(self):
        grid = [UNKNOWN] * N_CELLS
        grid[idx(5, 3)] = HIT
        grid[idx(5, 4)] = HIT
        for seed in range(20):
            self.assertIn(hunt(grid, [3], random.Random(seed)), {idx(5, 2), idx(5, 5)})

    def test_pd_prefers_centre_on_empty_board(self):
        grid = [UNKNOWN] * N_CELLS
        score = pd_scores(grid, list(FLEET), 0.0)
        self.assertGreater(score[idx(4, 4)], score[idx(0, 0)])

    def test_mc_only_returns_unknown_cells(self):
        grid = [UNKNOWN] * N_CELLS
        grid[idx(3, 3)] = HIT
        for seed in range(10):
            self.assertEqual(grid[mc_det(grid, list(FLEET), random.Random(seed))], UNKNOWN)


class LatticeTests(unittest.TestCase):
    def test_lattice_hits_every_five_run(self):
        lattice = set(ORDER_LATTICE[:20])
        self.assertEqual(len(lattice), 20)
        for r in range(SIZE):
            for c in range(SIZE - 4):
                self.assertTrue(any(idx(r, c + k) in lattice for k in range(5)))
        for c in range(SIZE):
            for r in range(SIZE - 4):
                self.assertTrue(any(idx(r + k, c) in lattice for k in range(5)))


if __name__ == "__main__":
    unittest.main()
