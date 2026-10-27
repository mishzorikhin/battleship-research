import random
import unittest

from battleship.engine import FLEET, N_CELLS, SIZE, UNKNOWN, HIT, play, rc, idx
from battleship.placements import WEIGHTS, random_layout
from battleship.strategies import ATTACKERS, ORDER_LATTICE, hunt, pd_scores, mc_det


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


class GameTests(unittest.TestCase):
    def test_every_attacker_finishes_without_repeats(self):
        for name, attacker in ATTACKERS.items():
            for seed in range(5):
                layout = random_layout(random.Random(f"L{seed}"), "random")
                shots, log = play(layout, attacker.choose, random.Random(f"A{seed}"), record=True)
                cells = [i for i, _ in log]
                self.assertEqual(len(cells), len(set(cells)), name)
                self.assertLessEqual(shots, N_CELLS, name)
                self.assertEqual(sum(1 for _, o in log if o != "miss"), sum(FLEET), name)

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
