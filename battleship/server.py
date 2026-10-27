"""Веб-интерфейс на стандартной библиотеке: реплей партии и запуск турниров."""
import json
import os
import random
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .engine import SIZE, N_CELLS, play, rc
from .placements import PLACEMENT_NAMES, fleet_rng, random_layout
from .strategies import ATTACKERS, ATTACKER_ORDER
from .tournament import load, run_tournament, save, summarize

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LATEST = os.path.join(RESULTS_DIR, "tournament.json")
MAX_GAMES = 5000

JOBS = {}
JOBS_LOCK = threading.Lock()


def _cells(indices):
    return [list(rc(i)) for i in indices]


def replay(attacker, placement, seed):
    layout = random_layout(fleet_rng(placement, seed), placement)
    rng = random.Random(f"attack|{attacker}|{placement}|{seed}")
    shots, log = play(layout, ATTACKERS[attacker].choose, rng, record=True)
    return {
        "attacker": attacker,
        "placement": placement,
        "seed": seed,
        "shots": shots,
        "ships": [_cells(s) for s in layout.ships],
        "moves": [[*rc(i), outcome] for i, outcome in log],
    }


def build_summary(payload):
    """Сводка по каждой паре (тактика, расстановка) + сравнение с контролем."""
    out = {}
    for a, by_placement in payload["results"].items():
        out[a] = {}
        for p, values in by_placement.items():
            out[a][p] = summarize(values)
    return out


def _start_job(attackers, placements, games, offset):
    job_id = uuid.uuid4().hex[:10]
    job = {"id": job_id, "state": "running", "done": 0, "total": len(attackers) * len(placements) * games}
    with JOBS_LOCK:
        JOBS[job_id] = job

    def progress(done, total):
        job["done"], job["total"] = done, total

    def work():
        try:
            results = run_tournament(attackers, placements, games, progress=progress, offset=offset)
            os.makedirs(RESULTS_DIR, exist_ok=True)
            path = LATEST if offset == 0 else os.path.join(RESULTS_DIR, f"tournament_{job_id}.json")
            save(results, path, games, offset)
            job["state"] = "done"
            job["file"] = os.path.relpath(path, os.path.dirname(WEB_DIR))
        except Exception as exc:  # показываем причину в UI, а не роняем сервер
            job["state"] = "error"
            job["error"] = repr(exc)

    threading.Thread(target=work, daemon=True).start()
    return job


class Handler(BaseHTTPRequestHandler):
    server_version = "Battleship/1.0"

    def log_message(self, fmt, *args):  # тихий лог
        pass

    def _json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, content_type):
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json({"error": message}, status)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self._file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
            if url.path == "/api/meta":
                return self._json({
                    "board": SIZE,
                    "fleet": [5, 4, 3, 3, 2],
                    "attackers": [
                        {"name": n, "title": ATTACKERS[n].title, "description": ATTACKERS[n].description}
                        for n in ATTACKER_ORDER
                    ],
                    "placements": [{"name": k, "title": v} for k, v in PLACEMENT_NAMES.items()],
                    "max_games": MAX_GAMES,
                })
            if url.path == "/api/replay":
                attacker = q.get("attacker", "pd_hint")
                placement = q.get("placement", "random")
                seed = int(q.get("seed", "0"))
                if attacker not in ATTACKERS or placement not in PLACEMENT_NAMES:
                    return self._error(HTTPStatus.BAD_REQUEST, "неизвестная тактика или расстановка")
                return self._json(replay(attacker, placement, seed))
            if url.path.startswith("/api/job/"):
                job_id = url.path.rsplit("/", 1)[-1]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if not job:
                    return self._error(HTTPStatus.NOT_FOUND, "задача не найдена")
                return self._json(job)
            if url.path == "/api/results":
                if not os.path.exists(LATEST):
                    return self._json({"payload": None})
                payload = load(LATEST)
                return self._json({"payload": payload, "summary": build_summary(payload)})
            return self._error(HTTPStatus.NOT_FOUND, "нет такой страницы")
        except (ValueError, KeyError) as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/tournament":
            return self._error(HTTPStatus.NOT_FOUND, "нет такой страницы")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            attackers = body.get("attackers") or list(ATTACKER_ORDER)
            placements = body.get("placements") or list(PLACEMENT_NAMES)
            games = int(body.get("games", 200))
            offset = int(body.get("offset", 0))
        except (ValueError, json.JSONDecodeError) as exc:
            return self._error(HTTPStatus.BAD_REQUEST, f"плохой запрос: {exc}")
        if any(a not in ATTACKERS for a in attackers):
            return self._error(HTTPStatus.BAD_REQUEST, "неизвестная тактика")
        if any(p not in PLACEMENT_NAMES for p in placements):
            return self._error(HTTPStatus.BAD_REQUEST, "неизвестная расстановка")
        if not 1 <= games <= MAX_GAMES:
            return self._error(HTTPStatus.BAD_REQUEST, f"games должно быть от 1 до {MAX_GAMES}")
        job = _start_job(attackers, placements, games, offset)
        return self._json(job, HTTPStatus.ACCEPTED)


def serve(host="127.0.0.1", port=8000):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Battleship research UI: http://{host}:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
