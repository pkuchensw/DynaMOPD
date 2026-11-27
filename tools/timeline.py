"""Build a self-contained physical GPU timeline from one job's receipts and NVML samples."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def records(path):
    # A live reader can encounter the partial last write of a line-buffered writer.
    with path.open("rb") as file:
        for line in file:
            try:
                yield json.loads(line)
            except (ValueError, UnicodeError):
                if line.endswith(b"\n"):
                    raise
                return


def report(folder):
    groups, rows, rounds, pending, warnings = {}, [], [], {}, []
    config, backend, status = {}, "", "Awaiting run completion"
    last = 0.
    simulated = False

    def add(indices, start, end, kind, detail):
        if end > start:
            for i in indices:
                rows[i]["spans"].append([start, end, kind, detail])

    for e in records(folder / "events.jsonl"):
        event = e["event"]
        last = max(last, e.get("t", 0.))
        if event == "config":
            backend, config = e["backend"], e["config"]
            # Keep historical traces readable after moving to the built-in simulator.
            simulated = backend in ("simulation", "simrl")
        elif event == "executor_ready":
            for p in e["placements"]:
                group = p["group"]
                groups[group] = []
                for d in p["devices"]:
                    groups[group].append(len(rows))
                    rows.append(dict(node=p["node"], device=str(d), group=group, spans=[], samples=[]))
        elif event == "round_start":
            rounds.append(dict(version=e["version"], start=e["t"], end=None))
        elif event == "round_end" and rounds:
            rounds[-1]["end"] = e["t"]
        elif event == "run_failed":
            status = "Failed"
        elif event == "run_finished":
            status = "Finished"
        elif event == "dispatch":
            pending[e["gpu"]] = (e["t"], e.get("load_version"), e["op"], e.get("peer"))
        elif event == "complete" and groups:
            gpu, op = e["gpu"], e["op"]
            a, b = e["start"], e["end"]
            dispatch = pending.pop(gpu, None)
            primary = groups[gpu]
            peer = e.get("peer")
            other = groups[peer] if peer is not None else []
            detail = f"v{e['version']} · group {gpu}"
            if other:
                detail += f" ↔ {peer}"
            if op == "switch":
                new = e.get("new", "?")
                published = dispatch and dispatch[1] is not None and dispatch[1] > e["version"]
                detail += f" · {e.get('old', e.get('role', '?'))} → {new}"
                if published:
                    detail += f" · v{dispatch[1]}"
                add(primary, a, b, "publish" if published or e.get("tensor_publish_s") else "switch", detail)
            elif op == "move" and "export_start" in e:
                for name, ids in (("export", primary), ("install", other), ("release", primary)):
                    if name + "_start" in e:
                        add(ids, e[name+"_start"], e[name+"_end"], "move", detail+" · "+name)
            elif op in ("decode", "score"):
                ids = primary + other
                sizes = e.get("replica_requests")
                width = e.get("inference_tp_size", e.get("tp_width", len(ids)))
                if op == "score":
                    detail += " · " + str(e.get("role", "teacher"))
                if peer is None and sizes and len(sizes) * width == len(ids):
                    for lane, count in enumerate(sizes):
                        add(ids[lane*width:(lane+1)*width], a, b,
                            op if count else "empty", f"{detail} · replica {lane} · {count} req")
                else:
                    add(ids, a, b, op, detail+" · GPU group receipt")
            else:
                add(primary + other, a, b, op, detail)
    if status == "Awaiting run completion" and rounds and all(r["end"] is not None for r in rounds) and len(rounds) == config.get("rounds"):
        status = "All rounds committed"
    # NVML IDs can repeat on different nodes. Each allocation writes its own file.
    by_device = {(r["node"], r["device"]): r for r in rows}
    for file in sorted((folder / "gpu-telemetry").glob("*.jsonl")):
        it = records(file)
        header = next(it, None)
        if not header:
            continue
        keys = [(header["node"], str(d)) for d in header["devices"]]
        for rec in it:
            if "error" in rec:
                warnings.append(f"{header['node']}: {rec['error']}")
                continue
            last = max(last, rec["t"])
            for key, (util, memory) in zip(keys, rec["values"], strict=True):
                if key in by_device:
                    row = by_device[key]
                    row["interval"] = header["interval"]
                    row["clock_error_s"] = header.get("clock_error_s", 0.)
                    row["samples"].append([round(rec["t"], 4), util,
                                           None if memory is None else round(memory / 2**30, 3)])
    for gpu, (since, _, op, peer) in pending.items():
        add(groups.get(gpu, []) + groups.get(peer, []), since, last, "pending", op+" · dispatched, awaiting completion")
    for row in rows:
        row["spans"].sort(key=lambda s: s[0])
    metrics = list(records(folder / "metrics.jsonl")) if (folder / "metrics.jsonl").exists() else []
    return dict(name=folder.name, backend=backend, simulated=simulated, status=status, rows=rows,
                rounds=rounds, end=last, metrics=metrics, warnings=warnings)


def render(folder, live=False):
    data = json.dumps(report(folder), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # User-selected node names, routes and directory names must remain inert in HTML.
    data = data.replace("&", r"\u0026").replace("<", r"\u003c").replace(">", r"\u003e")
    return Path(__file__).with_name("timeline.html").read_text(encoding="utf-8").replace("__LIVE__", str(live).lower()).replace("__DATA__", data)


def write_timeline(folder, output=None):
    folder = Path(folder)
    output = Path(output) if output else folder / "gpu-timeline.html"
    html = render(folder)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as f:
        try:
            f.write(html)
            f.close()
            os.replace(f.name, output)
        finally:
            Path(f.name).unlink(missing_ok=True)
    return output


def serve(folder, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/data.json"):
                self.send_error(404)
                return
            try:
                if self.path == "/data.json":
                    page = json.dumps(report(folder), allow_nan=False, separators=(",", ":")).encode()
                    content_type = "application/json"
                else:
                    page = render(folder, live=True).encode()
                    content_type = "text/html; charset=utf-8"
            except (OSError, ValueError) as exc:
                self.send_error(503, "Timeline unavailable")
                self.log_error("%s", exc)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as http:
        print(f"http://127.0.0.1:{http.server_port}/", flush=True)
        http.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.serve:
        serve(args.run, args.port)
    else:
        print(write_timeline(args.run, args.output))


if __name__ == "__main__":
    main()
