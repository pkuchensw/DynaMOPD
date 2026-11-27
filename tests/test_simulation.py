"""Run with: python -S -m unittest discover -s tests -p test_simulation.py."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "DynaMOPD", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PACKAGE = importlib.util.module_from_spec(PACKAGE_SPEC)
sys.modules["DynaMOPD"] = PACKAGE
PACKAGE_SPEC.loader.exec_module(PACKAGE)

from DynaMOPD.backends.sim_model import KvCacheManager, Request, ScoringCost, UnifiedRoofline
from DynaMOPD.tools.audit import audit
from DynaMOPD.tools.timeline import report


class SimulationModelTests(unittest.TestCase):
    def test_kv_growth_failure_preserves_existing_allocations(self):
        cache = KvCacheManager(3)
        first = Request(1, 16, 32, 0)
        second = Request(2, 16, 32, 0)
        self.assertIsNotNone(cache.allocate(first, 16))
        first.prefilled = 16
        self.assertIsNotNone(cache.allocate(second, 16))
        second.prefilled = 16
        self.assertEqual((cache.num_used_blocks, cache.num_free_blocks), (2, 1))

        # Growing across two page boundaries cannot consume the last free page.
        self.assertIsNone(cache.allocate(first, 17))
        self.assertEqual(cache.blocks_held(first), 1)
        self.assertEqual(cache.blocks_held(second), 1)
        self.assertEqual((cache.num_used_blocks, cache.num_free_blocks), (2, 1))
        self.assertEqual(first.prefilled, 16)

        self.assertIsNotNone(cache.allocate(first, 1))
        first.prefilled = 17
        self.assertEqual((cache.num_used_blocks, cache.num_free_blocks), (3, 0))
        # The rest of this page does not require more capacity.
        self.assertIsNotNone(cache.allocate(first, 15))
        first.prefilled = 32
        self.assertEqual(cache.blocks_held(first), 2)
        cache.free(first)
        cache.free(first)
        cache.free(second)
        self.assertEqual((cache.num_used_blocks, cache.num_free_blocks), (0, 3))

    def test_kv_managers_keep_independent_rank_capacity(self):
        left, right = KvCacheManager(2), KvCacheManager(2)
        request = Request(1, 32, 32, 0)
        self.assertIsNotNone(left.allocate(request, 32))
        self.assertEqual((left.num_free_blocks, right.num_free_blocks), (0, 2))
        self.assertIsNotNone(right.allocate(request, 32))
        request.prefilled = 32
        left.free(request)
        self.assertEqual((left.num_used_blocks, right.num_used_blocks), (0, 2))
        right.free(request)
        self.assertEqual(right.num_free_blocks, 2)

    def test_roofline_prefill_and_decode_costs(self):
        model = UnifiedRoofline(F=0.01, W=0.003, G=0.001, A_p=0.0002, A_d=0.0001)
        self.assertEqual(model.step_time(0, 0, 0, 0), 0)
        self.assertAlmostEqual(model.step_time(1, 0, 1, 1), 0.01)
        # Two prompts of lengths 4 and 6: 10 tokens, squared lengths 16 + 36.
        self.assertAlmostEqual(model.step_time(10, 52, 0, 2), 0.0182)
        self.assertGreater(model.step_time(2, 0, 200, 2), model.step_time(2, 0, 20, 2))

    def test_roofline_decode_coefficients_match_growing_contexts(self):
        model = UnifiedRoofline(F=0.01, W=0.003, G=0.001, A_d=0.0001)
        batch, context = 3, 12
        floor, initial, slope = model.decode_coeffs(batch, context)
        for step in range(30):
            self.assertAlmostEqual(max(floor, initial + step * slope),
                                   model.step_time(batch, 0, context + step * batch, batch))

    def test_scoring_charges_actual_tokens_and_one_launch(self):
        model = ScoringCost(launch_s=0.2, token_s=0.03)
        self.assertEqual(model.predict(()), 0)
        self.assertAlmostEqual(model.predict((2, 5)), 0.41)
        self.assertAlmostEqual(model.predict((7,)), model.predict((2, 5)))


class SimulationCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dynamopd-simulation-test-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def config(self, **overrides):
        cfg = json.loads((ROOT / "configs/minimal-4gpu.json").read_text(encoding="utf-8"))
        cfg.update(overrides)
        return cfg

    def run_cli(self, cfg, name, *extra):
        config = self.folder / (name + ".json")
        config.write_text(json.dumps(cfg), encoding="utf-8")
        output = self.folder / name
        # Load this checkout as its declared package even when a source archive
        # was extracted into a directory such as dynamopd-0.1.0. No installation,
        # site-packages, or caller PYTHONPATH is available to the CLI process.
        bootstrap = (
            "import importlib.util, runpy, sys\n"
            f"root = {str(ROOT)!r}\n"
            "spec = importlib.util.spec_from_file_location('DynaMOPD', "
            "root + '/__init__.py', submodule_search_locations=[root])\n"
            "package = importlib.util.module_from_spec(spec)\n"
            "sys.modules['DynaMOPD'] = package\n"
            "spec.loader.exec_module(package)\n"
            "runpy.run_module('DynaMOPD.run', run_name='__main__')\n"
        )
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        env["PYTHONIOENCODING"] = "utf-8"
        command = [sys.executable, "-S", "-c", bootstrap, "--simulate",
                   "--config", str(config), "--out", str(output), *map(str, extra)]
        process = subprocess.run(command, cwd=self.folder, env=env, capture_output=True,
                                 text=True, encoding="utf-8", timeout=60)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events[0]["backend"], "simulation")
        self.assertEqual(events[-1]["event"], "run_finished")
        self.assertTrue((output / "gpu-timeline.html").is_file())
        return output, events, audit(output)

    def replay(self, name, lengths, *, rounds=1, cap=128):
        folder = self.folder / name
        folder.mkdir()
        for version in range(rounds):
            rows = [dict(rid=f"v{version}/r{i}", seed=1901 + version * len(lengths) + i,
                         teacher="math" if i % 2 == 0 else "general", prompt=[1] * (i + 2),
                         response=[1] * length, finish="length" if length == cap else "eos",
                         max_new_tokens=cap)
                    for i, length in enumerate(lengths)]
            (folder / f"samples-v{version}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return folder

    def test_all_fixed_width_policies_complete_two_audited_rounds(self):
        for policy in ("natural", "merge", "cost_search"):
            with self.subTest(policy=policy):
                output, events, checked = self.run_cli(self.config(policy=policy), policy)
                self.assertEqual([r["published"] for r in checked["rounds"]], [1, 2])
                for row in checked["rounds"]:
                    self.assertEqual(row["generated_tokens"], 676)
                    self.assertEqual(row["training_tokens"], 676)
                    self.assertEqual((row["eos"], row["truncated"], row["trained"]), (4, 4, 8))
                barriers = [e for e in events if e["event"] == "rollout_barrier"]
                self.assertEqual([e["loaded_version"] for e in barriers], [1, 2])
                timeline = report(output)
                self.assertTrue(timeline["simulated"])
                self.assertEqual(timeline["status"], "Finished")
                self.assertEqual(len(timeline["rows"]), 4)

    def test_replay_preserves_recorded_lengths_limits_and_ids(self):
        lengths = [3, 8, 20, 32]
        replay = self.replay("recorded", lengths, rounds=2, cap=32)
        cfg = self.config(batch_size=4, max_new_tokens=32, policy="cost_search")
        _, events, checked = self.run_cli(cfg, "replayed", "--replay", replay)
        for version, row in enumerate(checked["rounds"]):
            self.assertEqual(row["generated_tokens"], sum(lengths))
            self.assertEqual(row["training_tokens"], sum(lengths))
            self.assertEqual((row["eos"], row["truncated"]), (3, 1))
            final = {e["rid"]: e for e in events
                     if e["event"] == "prefix" and e["version"] == version and e["finish"]}
            self.assertEqual({rid: e["end"] for rid, e in final.items()},
                             {f"v{version}/r{i}": n for i, n in enumerate(lengths)})

    def test_elastic_tp_expands_and_releases_pages_in_both_models(self):
        replay = self.replay("tail", [8, 128])
        for kind in ("reshard", "standby"):
            with self.subTest(kind=kind):
                cfg = self.config(rounds=1, batch_size=2, quantum=8, policy="elastic_tp",
                                  max_train_groups=1, phase_barriers=True)
                cfg["costs"].update(switch=0.001, score_token=0.00001, train_token=0.00001)
                if kind == "reshard":
                    cfg["inference_tp_size"] = 2
                    profile = json.loads((ROOT / "configs/sim-tail-tp2-tp4.json").read_text())
                    profile.update(communicator_s=0, enter_s=0, exit_s=0,
                                   teacher_to_student_s=0, student_to_teacher_s=0)
                    profile["roofline_by_tp"] = {"2": dict(F=0.02, W=0, G=0),
                                                 "4": dict(F=0.001, W=0, G=0)}
                    cfg["sim_tp"] = profile
                else:
                    cfg["rollout_tp_expansion"] = dict(source_tp=1, target_tp=4,
                        narrow_token_s=0.02, wide_token_s=0.001, enter_s=0.001,
                        exit_s=0.001, prefill_token_s=0.000001)
                _, events, checked = self.run_cli(cfg, kind, "--replay", replay)
                spans = [e for e in events if e["event"] == "complete"]
                self.assertTrue(any(e["op"] == "tp_expand" for e in spans))
                self.assertTrue(any(e["op"] == "decode" and e.get("peer") is not None for e in spans))
                shrinks = [e for e in spans if e["op"] == "tp_shrink"]
                self.assertTrue(shrinks)
                self.assertTrue(all(e["peer_kv_used_blocks"] == 0 for e in shrinks))
                self.assertEqual(checked["rounds"][0]["training_tokens"], 136)

    def test_timeline_recognizes_new_and_historical_simulation_traces(self):
        for backend, simulated in (("simulation", True), ("simrl", True), ("sglang_fsdp2", False)):
            with self.subTest(backend=backend):
                folder = self.folder / backend
                folder.mkdir()
                events = [dict(event="config", backend=backend, config={}), dict(event="run_finished")]
                (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
                self.assertEqual(report(folder)["simulated"], simulated)


if __name__ == "__main__":
    unittest.main()
