"""Run E1–E4 in order, resuming caches and protecting validation-selected policies."""

import argparse
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from routing_project.config import environment, load_config, run_location
from routing_project.prepare_data import verify_inputs
from routing_project.storage import (
    file_hash,
    manifest_path,
    read_json,
    timestamp,
    verify_manifest,
    write_json,
    writer_lock,
)


def gpu_processes(gpu_uuid):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    return [
        {"gpu_uuid": fields[0].strip(), "pid": int(fields[1])}
        for line in output.splitlines()
        if len(fields := line.split(",")) == 2 and fields[0].strip() == gpu_uuid
    ]


def selected_gpu():
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not device or "," in device:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one allocated GPU index or UUID")
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--id", device, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    if len(gpu_uuid.splitlines()) != 1:
        raise RuntimeError("Expected one selected GPU")
    if processes := gpu_processes(gpu_uuid):
        raise RuntimeError(f"Selected GPU is occupied: {processes}")
    # The check runs in a child so the launcher holds no CUDA context during E4.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "assert torch.cuda.is_bf16_supported(), 'BF16 support required'; "
            "print({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__})",
        ],
        check=True,
    )
    return gpu_uuid


def run_latency(command, gpu_uuid, report_path):
    if processes := gpu_processes(gpu_uuid):
        raise RuntimeError(f"E4 requires a quiet GPU: {processes}")
    monitoring = {"gpu_uuid": gpu_uuid, "started_at": timestamp(), "samples": [], "errors": []}
    stop = threading.Event()
    process = subprocess.Popen(command)

    def monitor():
        while not stop.is_set():
            try:
                others = [row for row in gpu_processes(gpu_uuid) if row["pid"] != process.pid]
                monitoring["samples"].append({"at": timestamp(), "other_processes": others})
                if others:
                    if process.poll() is None:
                        process.terminate()
                    break
            except Exception as exc:
                monitoring["errors"].append(str(exc))
                if process.poll() is None:
                    process.terminate()
                break
            stop.wait(5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        code = process.wait()
    finally:
        stop.set()
        if process.poll() is None:
            process.terminate()
            process.wait()
        thread.join()
        monitoring["finished_at"] = timestamp()
        monitoring["uncontended_in_samples"] = not monitoring["errors"] and not any(
            row["other_processes"] for row in monitoring["samples"]
        )
        write_json(report_path, monitoring)
    if code or not monitoring["uncontended_in_samples"]:
        raise RuntimeError("E4 interrupted or GPU contention detected; retry with a new --session name")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--experiments", choices=["all", "e1-e3", "e4"], default="all")
    parser.add_argument("--session", default="session-1", help="Name for this E4 timing allocation")
    args = parser.parse_args()
    if not args.session.replace("-", "").replace("_", "").isalnum():
        parser.error("Session must contain only letters, digits, hyphens, or underscores")
    config = load_config(args.config)
    root = Path(config["_root"])
    os.chdir(root)
    verify_inputs(config)
    output = run_location(config, "results")
    state_path = output / "progress.json"
    state = {"run_id": config["run_id"], "session": args.session, "completed_stages": []}

    def record(stage, status):
        state.update(stage=stage, status=status, updated_at=timestamp())
        if status == "complete" and stage not in state["completed_stages"]:
            state["completed_stages"].append(stage)
        write_json(state_path, state)
        print({"stage": stage, "status": status}, flush=True)

    def command(module, *arguments):
        return [
            sys.executable,
            "-u",
            "-m",
            f"routing_project.{module}",
            "--config",
            config["_path"],
            *arguments,
        ]

    def run(stage, module, *arguments):
        record(stage, "running")
        subprocess.run(command(module, *arguments), check=True)
        record(stage, "complete")

    with writer_lock(state_path):
        software = environment()
        execution = {
            "run_id": config["run_id"],
            "software": software,
            "sources": {
                str(p.relative_to(root)): file_hash(p)
                for p in sorted((root / "routing_project").glob("*.py"))
            },
        }
        execution["sources"]["scripts/run_experiments.py"] = file_hash(Path(__file__))
        execution_path = output / "execution.json"
        if execution_path.exists():
            previous = read_json(execution_path)
            if any(previous.get(key) != value for key, value in execution.items()):
                raise ValueError("Code or environment changed; use a fresh results/data/artifacts location")
        else:
            write_json(execution_path, execution)
        if state_path.exists():
            state["completed_stages"] = read_json(state_path)["completed_stages"]

        def interrupted(signum, frame):
            raise SystemExit(128 + signum)

        handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            if manifest_path(config).exists():
                verify_manifest(config)
            elif args.experiments == "e4":
                raise ValueError("Run E1–E3 to train and freeze the policies before E4")
            gpu_uuid = selected_gpu()
            if args.experiments in ("all", "e1-e3"):
                run("input_check", "audit")
                if not manifest_path(config).exists():
                    run("training_generation", "run_modes", "--split", "train", "--modes", "nonthinking")
                    run(
                        "validation_generation",
                        "run_modes",
                        "--split",
                        "validation",
                        "--modes",
                        "nonthinking",
                        "thinking",
                    )
                    run("train_validation_features", "extract_features", "--splits", "train", "validation")
                    run("probe_training", "train_probes")
                    run("policy_freeze", "select_policies")
                run("test_generation", "run_modes", "--split", "test", "--modes", "nonthinking", "thinking")
                run("test_features", "extract_features", "--splits", "test")
                run("formal_audit", "audit", "--formal")
                run("e1_e3", "evaluate")
                run("figures", "make_figures")
            if args.experiments in ("all", "e4"):
                verify_manifest(config)
                record("e4", "running")
                run_latency(
                    command("benchmark_latency", "--session", args.session),
                    gpu_uuid,
                    output / "e4_sessions" / f"{args.session}_gpu_monitor.json",
                )
                record("e4", "complete")
                run("figures", "make_figures")
            record("finished", "complete")
            print({"results": str(output)}, flush=True)
        except BaseException as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            record(state.get("stage", "initialization"), "failed")
            raise
        finally:
            signal.signal(signal.SIGTERM, handler)


if __name__ == "__main__":
    main()
