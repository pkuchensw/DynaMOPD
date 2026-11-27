"""CLI for the same trainer used by programmatic jobs."""
import argparse
import json
from pathlib import Path

from .cluster.topology import describe_layout, node_shape
from .trainer import fit

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/sglang-fsdp2-h20.json")
    parser.add_argument("--data", type=Path, default=ROOT/"data/native-smoke.jsonl")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-layout", action="store_true", help="print the planned multi-node reservation offline and exit")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--algorithm", help="built-in objective name or package.module:Class")
    parser.add_argument("--policy", help="cost_search, natural, merge, elastic_tp, or package.module:Class")
    parser.add_argument("--rollout-tp-profile",type=Path,help="local standby TP topology; simulation also reads timings, SGLang measures them on the allocation")
    parser.add_argument("--sim-tp-profile", type=Path, help="TP rooflines and resize costs for simulation")
    parser.add_argument("--cluster-config", type=Path, help="JSON cluster section overriding config.cluster")
    parser.add_argument("--groups", type=int, help="total logical GPU groups across all nodes")
    parser.add_argument("--max-train-groups", type=int, help="maximum concurrent FSDP2 training groups")
    parser.add_argument("--inference-tp-size", type=int, help="SGLang tensor-parallel width within each FSDP group")
    parser.add_argument("--phase-barriers", action=argparse.BooleanOptionalAction, default=None,
                        help="finish generation before scoring and finish scoring before training")
    parser.add_argument("--track", nargs="+", choices=["wandb", "swanlab"], help="optional experiment trackers")
    parser.add_argument("--tracking-mode", choices=["offline", "online"], help="default: offline")
    parser.add_argument("--tracking-project")
    parser.add_argument("--run-name")
    parser.add_argument("--gpu-sample-interval", type=float, help="NVML sample interval in seconds; 0 disables sampling")
    parser.add_argument("--timeline", action=argparse.BooleanOptionalAction, default=None,
                        help="write the self-contained HTML timeline when the job exits (default: on)")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--replay", type=Path, help="simulation workload from samples-v*.jsonl")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--resume-version", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.simulate and args.policy is None and "sim_policy" in config:
        config["policy"] = config["sim_policy"]
    if args.rollout_tp_profile:
        config["rollout_tp_expansion"] = json.loads(args.rollout_tp_profile.read_text())
    if args.sim_tp_profile:
        if not args.simulate:
            parser.error("--sim-tp-profile requires --simulate")
        config["sim_tp"] = json.loads(args.sim_tp_profile.read_text())
    for key in ("algorithm", "policy", "groups", "max_train_groups", "inference_tp_size", "phase_barriers", "rounds",
                "track", "tracking_mode", "tracking_project", "run_name", "gpu_sample_interval", "timeline"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.cluster_config:
        config["cluster"] = json.loads(args.cluster_config.read_text())
        shape = node_shape(config["cluster"])
        if shape and args.groups is None:
            config["groups"] = sum(count for _, count in shape)
    if args.check_layout:
        print(json.dumps(describe_layout(config), indent=2))
        return
    if args.out is None:
        parser.error("--out is required to run training or simulation")
    fit(config, output=args.out, data_path=args.data, state_dir=args.state_dir, simulate=args.simulate,
        replay=args.replay, resume_version=args.resume_version,
        on_round=lambda report: print(json.dumps(report), flush=True))


if __name__ == "__main__":
    main()
