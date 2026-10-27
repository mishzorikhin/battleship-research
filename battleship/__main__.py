"""Командная строка: python -m battleship {serve,tournament,report}."""
import argparse
import os
import time

from .placements import PLACEMENT_NAMES
from .server import LATEST, RESULTS_DIR, serve
from .strategies import ATTACKER_ORDER
from .tournament import load, run_tournament, save


def _tournament(args):
    attackers = args.attackers or list(ATTACKER_ORDER)
    placements = args.placements or list(PLACEMENT_NAMES)
    start = time.perf_counter()

    def progress(done, total):
        print(f"\r{done}/{total} партий, {time.perf_counter() - start:6.1f} с", end="", flush=True)

    results = run_tournament(attackers, placements, args.games, workers=args.workers,
                             progress=progress, offset=args.offset)
    print()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save(results, args.out, args.games, args.offset)
    print(f"сохранено: {args.out}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="battleship")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="веб-интерфейс")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_t = sub.add_parser("tournament", help="пакетный прогон без UI")
    p_t.add_argument("--games", type=int, default=500)
    p_t.add_argument("--offset", type=int, default=0, help="первый seed (для дозапуска)")
    p_t.add_argument("--attackers", nargs="*", choices=ATTACKER_ORDER)
    p_t.add_argument("--placements", nargs="*", choices=list(PLACEMENT_NAMES))
    p_t.add_argument("--workers", type=int, default=None)
    p_t.add_argument("--out", default=LATEST)

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        serve(args.host, args.port)
    elif args.cmd == "tournament":
        _tournament(args)


if __name__ == "__main__":
    main()
