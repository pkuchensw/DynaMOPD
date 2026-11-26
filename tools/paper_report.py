"""Render the comparison tables and GPU timelines of a local paper-data archive."""
import argparse
import base64
import csv
import gzip
import hashlib
import html
import io
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from urllib.parse import quote
from zipfile import ZipFile


SYSTEMS = ("dynamopd", "kdflow", "verl")
SYSTEM_NAMES = {"dynamopd": "DynaMOPD", "kdflow": "KDFlow", "verl": "verl"}
ABLATIONS = {
    "abbarriers": "阶段间设屏障", "abfixed": "关闭 TP 扩容", "abnomerge": "关闭请求合并",
    "abprefix": "关闭前缀评分", "abtrain": "关闭提前训练",
}


def csv_rows(archive, path):
    with archive.open(path) as source, io.TextIOWrapper(source, encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def legacy_scaling(key):
    return re.search(r"(?:^|/)scaling/\d+(?:/|$)", key) is not None


def case_family(key):
    return re.sub(r"-(?:h\d+|r\d+|s\d+)$", "", key)


def comparisons(archive):
    configs = json.loads(archive.read("configs/cases.json"))
    names = {row["case"]: row["name"] for row in csv_rows(archive, "tables/paired-matrix.csv")}
    systems = defaultdict(dict)
    for row in csv_rows(archive, "tables/comparisons.csv"):
        if not legacy_scaling(row["case"]):
            systems[row["case"]][row["system"]] = row
    groups = {case: [row["run_id"] for row in runs.values()] for case, runs in systems.items()}
    families = defaultdict(list)
    for case, runs in systems.items():
        if all(s in runs for s in SYSTEMS):
            families[case_family(case)].append({
                "key": case, "seed": int(runs["dynamopd"]["seed"]),
                "sec": {s: float(runs[s]["seconds_per_step"]) for s in SYSTEMS},
            })
    ordered = list(dict.fromkeys([*names, *families]))
    scenarios = []
    previous_section = None
    for family in ordered:
        if family not in families:
            continue
        members = sorted(families[family], key=lambda m: m["seed"])
        key = members[0]["key"]
        cfg = configs[key]
        gpus = int(systems[key]["dynamopd"]["gpus"])
        title = f"{gpus} 卡 · 首轮标定版" if family.startswith("measured-scaling/") else names.get(family, family)
        teachers = len(cfg["teachers"])
        section = (teachers, family.startswith("measured-scaling/"))
        scenarios.append(dict(id=family, title=title, gpus=gpus, count=teachers,
                              student=cfg["student"], batch=cfg["settings"]["batch_size"],
                              cap=cfg["settings"]["max_new_tokens"], members=members,
                              sec={s: mean(m["sec"][s] for m in members) for s in SYSTEMS},
                              sectionStart=bool(scenarios and section != previous_section)))
        previous_section = section
    ablations = []
    for row in csv_rows(archive, "ablation/paired_seed42.csv"):
        case = row["case"]
        if case in groups:
            source = systems[case]["dynamopd"]
            ablations.append(dict(key=case, name=ABLATIONS.get(case_family(case), case),
                                  value=float(row["ablation_seconds_over_full"]),
                                  config=f"{source['gpus']} 卡 · 全局 batch {source['batch']} · 种子 {source['seed']}"))
    return groups, scenarios, ablations


def paper_timelines(archive, wanted):
    runs, windows = {}, {}
    for row in csv_rows(archive, "tables/run_index.csv"):
        rid = row["run_id"]
        if rid in wanted:
            runs[rid] = dict(rid=rid, system=SYSTEM_NAMES.get(row["system"], row["system"]),
                             mode="gpu", basis=row["basis"], labels=[])
            windows[rid] = {}
    for row in csv_rows(archive, "timeline/lanes.csv"):
        if row["run_id"] in runs:
            labels = runs[row["run_id"]]["labels"]
            i = int(row["lane"])
            if len(labels) <= i:
                labels.extend([""] * (1+i-len(labels)))
            labels[i] = row["device"]
    for row in csv_rows(archive, "tables/per_step.csv"):
        if row["run_id"] in runs:
            windows[row["run_id"]][row["step"]] = dict(s=float(row["figure_interval_s"]), spans=[], actions=[], util={})
    for name in ("phases", "actions", "gpu_utilization"):
        for row in csv_rows(archive, f"timeline/{name}.csv"):
            if row["run_id"] not in runs:
                continue
            w = windows[row["run_id"]][row["step"]]
            if name == "gpu_utilization":
                w["util"].setdefault(row["lane"], []).append([float(row["t_s"]), float(row["util_pct"])])
                continue
            span = [float(row["start_s"]), float(row["end_s"]), [int(k) for k in row["lanes"].split("|") if k]]
            if name == "phases":
                w["spans"].append(span + [row["phase"], row["label"], ""])
            else:
                w["actions"].append(span + [row["action"], row["detail"], row["kind"]])
    return runs, windows


def render(archive, zip_url):
    groups, scenarios, ablations = comparisons(archive)
    wanted = {rid for ids in groups.values() for rid in ids}
    runs, windows = paper_timelines(archive, wanted)
    if "historical/timelines.json" in archive.namelist():
        history = json.loads(archive.read("historical/timelines.json"))
        for row in csv_rows(archive, "historical/run_index.csv"):
            case, rid = row["study"], row["run_id"]
            if legacy_scaling(case):
                continue
            entry = history[rid]
            runs[rid] = dict(rid=rid, system=row["system"], mode=row["mode"], basis=row["basis"], labels=entry["lanes"])
            windows[rid] = entry["windows"]
            groups.setdefault(case, []).append(rid)
    order = {rid: i for i, rid in enumerate(runs)}
    payloads, tags = set(), []
    for rid, run in runs.items():
        data = compact_json(windows[rid]).encode()
        key = hashlib.sha256(data).hexdigest()
        run.update(keys=list(windows[rid]), blob=key)
        if key not in payloads:
            blob = base64.b64encode(gzip.compress(data, compresslevel=6, mtime=0)).decode()
            tags.append(f'<script type="application/octet-stream" id="r{key}">{blob}</script>')
            payloads.add(key)
    index = dict(scenarios=scenarios, ablations=ablations, runs=list(runs.values()),
                 groups={k: [order[r] for r in ids] for k, ids in groups.items()})
    # Render the overview as ordinary HTML as well, so the results are readable without JavaScript.
    rows = []
    for case in scenarios:
        href = "#case/" + quote(case["members"][0]["key"], safe="")
        cells = [f'<td><a href="{href}">{html.escape(case["title"])}</a></td>', f'<td>{case["gpus"]}</td>']
        for system in ("kdflow", "verl"):
            speed = case["sec"][system] / case["sec"]["dynamopd"]
            style = "value slow" if speed < 1 else "value"
            cells.append(f'<td class="{style}">{speed:.2f}×</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    template = Path(__file__).with_name("paper_report.template.html").read_text(encoding="utf-8")
    return (template.replace("__ZIP_LINK__", html.escape(zip_url, quote=True))
            .replace("<!--HOME_ROWS-->", "".join(rows))
            .replace("/*INDEX*/", compact_json(index).replace("<", r"\u003c"))
            .replace("<!--PAYLOADS-->", "\n".join(tags)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Paper-data ZIP with comparison and timeline tables")
    parser.add_argument("--out", type=Path, help="Offline HTML output (default: next to archive)")
    args = parser.parse_args()
    output = args.out or args.archive.with_name("DynaMOPD-paper-report.html")
    if output.resolve() == args.archive.resolve():
        parser.error("output must differ from the input archive")
    with ZipFile(args.archive) as archive:
        doc = render(archive, quote(os.path.relpath(args.archive.resolve(), output.parent.resolve())))
    output.write_text(doc, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
