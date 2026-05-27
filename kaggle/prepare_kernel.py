"""
Prepare Kaggle kernel push directories for the currently authenticated Kaggle user.

The Kaggle CLI requires the pushed `kernel-metadata.json.id` to include the
owner slug, for example `USERNAME/nsa-phase1`. Source metadata in this repo
stays username-free; this helper writes generated metadata into /tmp for pushing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = Path("/tmp/nsa-kaggle-kernels")


@dataclass(frozen=True)
class PhaseKernel:
    source_dir: Path
    metadata_path: Path
    code_file: str
    slug: str


PHASES = {
    "phase0": PhaseKernel(
        source_dir=ROOT / "kaggle",
        metadata_path=ROOT / "kaggle" / "kernel-metadata.json",
        code_file="run_phase0.py",
        slug="nsa-phase0",
    ),
    "phase1": PhaseKernel(
        source_dir=ROOT / "kaggle" / "phase1",
        metadata_path=ROOT / "kaggle" / "phase1" / "kernel-metadata.json",
        code_file="run_phase1.py",
        slug="nsa-phase1",
    ),
    "phase2a": PhaseKernel(
        source_dir=ROOT / "kaggle" / "phase2a",
        metadata_path=ROOT / "kaggle" / "phase2a" / "kernel-metadata.json",
        code_file="run_phase2a.py",
        slug="nsa-phase2a",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set Kaggle kernel metadata owner to the active Kaggle CLI user."
    )
    parser.add_argument(
        "phase",
        choices=["phase0", "phase1", "phase2a", "all"],
        nargs="?",
        default="all",
        help="Which kernel metadata file to update.",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Override owner slug. Defaults to active Kaggle CLI username.",
    )
    parser.add_argument(
        "--hf-token-dataset",
        default=None,
        help=(
            "Optional private dataset source containing hf_token.txt. "
            "Use 'auto' for USERNAME/nsa-hf-token."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"Directory where generated push folders are written. Default: {DEFAULT_OUT_ROOT}",
    )
    args = parser.parse_args()

    username = args.username or detect_kaggle_username()
    hf_token_dataset = _resolve_token_dataset(args.hf_token_dataset, username)
    selected = PHASES.items() if args.phase == "all" else [(args.phase, PHASES[args.phase])]
    for phase, kernel in selected:
        output_dir = build_push_dir(
            phase=phase,
            kernel=kernel,
            username=username,
            hf_token_dataset=hf_token_dataset,
            out_root=args.out_root,
        )
        print(f"{phase}: {username}/{kernel.slug} -> {output_dir}")
    return 0


def detect_kaggle_username() -> str:
    env_username = os.environ.get("KAGGLE_USERNAME")
    if env_username:
        return env_username

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    kaggle_json = config_dir / "kaggle.json"
    if kaggle_json.exists():
        payload = json.loads(kaggle_json.read_text(encoding="utf-8"))
        username = payload.get("username")
        if username:
            return str(username)

    completed = subprocess.run(
        ["kaggle", "config", "view"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("- username:"):
            return stripped.split(":", 1)[1].strip()

    raise RuntimeError("Could not detect Kaggle username from env, kaggle.json, or CLI.")


def build_push_dir(
    phase: str,
    kernel: PhaseKernel,
    username: str,
    hf_token_dataset: str | None,
    out_root: Path,
) -> Path:
    output_dir = out_root / phase
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(kernel.source_dir / kernel.code_file, output_dir / kernel.code_file)

    payload = json.loads(kernel.metadata_path.read_text(encoding="utf-8"))
    payload["id"] = f"{username}/{kernel.slug}"
    payload["title"] = kernel.slug
    payload["code_file"] = kernel.code_file
    if hf_token_dataset:
        sources = list(payload.get("dataset_sources", []))
        if hf_token_dataset not in sources:
            sources.append(hf_token_dataset)
        payload["dataset_sources"] = sources
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def _resolve_token_dataset(value: str | None, username: str) -> str | None:
    if value is None:
        return None
    if value == "auto":
        return f"{username}/nsa-hf-token"
    return value


if __name__ == "__main__":
    raise SystemExit(main())
