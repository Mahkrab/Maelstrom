#!/usr/bin/env python3
"""Collect alternating Driver-only grid/sort samples; preserve every raw result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("ptx", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sort-executable", type=Path, help="optional unmodified control executable for complete-step comparisons")
    parser.add_argument("--sizes", nargs="+", type=int, default=[65536, 262144, 1048576, 2097152, 4194304])
    parser.add_argument("--distributions", nargs="+", choices=["normal", "dense", "sparse"], default=["normal", "dense", "sparse"])
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1, help="accepted warm-up steps in complete-step mode")
    parser.add_argument("--steps", type=int, help="use the complete Simulation example executable and collect this many measured steps after one warm-up")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"host": platform.node(), "platform": platform.platform(), "pairs": args.pairs,
                "sizes": args.sizes, "distributions": args.distributions, "warmup_steps": args.warmup,
                "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                "ptx_sha256": hashlib.sha256(args.ptx.read_bytes()).hexdigest(),
                "method": "one untimed discovery then one timed discovery and separate CUDA-event stages per process; five alternating pairs by default"}
    if args.sort_executable is not None:
        if args.steps is None:
            parser.error("--sort-executable requires --steps")
        metadata["sort_executable_sha256"] = hashlib.sha256(args.sort_executable.read_bytes()).hexdigest()
    if args.steps is not None:
        metadata["method"] = "one untimed accepted step then timed total-only Simulation steps; alternating policies; exact final-state hashes required"
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], capture_output=True, text=True, check=True)
    metadata["gpu"] = result.stdout
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    hashes = {}
    with (args.output / "samples.jsonl").open("w") as samples:
        for distribution in args.distributions:
            for size in args.sizes:
                for pair in range(args.pairs):
                    for policy in (["sort", "grid"] if pair % 2 == 0 else ["grid", "sort"]):
                        command = [str(args.executable.resolve()), str(args.ptx.resolve()), policy, str(size), distribution, "1"]
                        environment = os.environ.copy()
                        if args.steps is not None:
                            command = [str(args.executable.resolve()), str(size), distribution, str(args.steps)]
                            if policy == "sort" and args.sort_executable is not None:
                                command[0] = str(args.sort_executable.resolve())
                            environment["MAELSTROM_CUDA_SPATIAL_INDEX"] = policy
                        if args.steps is not None and args.warmup != 1:
                            command.append(str(args.warmup))
                        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
                        stem = f"{distribution}-{size}-{pair}-{policy}"
                        (args.output / f"{stem}.stdout").write_text(result.stdout)
                        (args.output / f"{stem}.stderr").write_text(result.stderr)
                        if result.returncode:
                            raise RuntimeError(f"{stem} exited {result.returncode}: {result.stderr}")
                        for line in result.stdout.splitlines():
                            record = json.loads(line)
                            record["pair"] = pair
                            if args.steps is not None:
                                key = (distribution, size)
                                expected = hashes.setdefault(key, record["accepted_state_sha256"])
                                if expected != record["accepted_state_sha256"]:
                                    raise RuntimeError(f"exact state mismatch: {stem}")
                            samples.write(json.dumps(record) + "\n")
                        samples.flush()
                print(f"completed {distribution}: {size}", flush=True)


if __name__ == "__main__":
    main()
