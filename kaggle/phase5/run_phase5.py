"""
Kaggle kernel script for Phase 5: Mixed CAA + Suppression Detection Training on T4 GPU.

Trains a fresh QLoRA adapter on mixed CAA + suppression-positive data, then evaluates
detection rates and safety. All output artifacts are written to /kaggle/working/.
"""

import os
import subprocess
import sys


PHASE2A_REQUIRED_FILES = {
    "caa_vectors.pt",
    "concept_order.json",
    "train_dataset.jsonl",
    "eval_dataset.jsonl",
}


def install_deps():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "transformers>=4.51", "datasets>=2.19",
        "huggingface-hub>=0.23", "pyyaml>=6.0",
        "bitsandbytes>=0.43", "accelerate>=0.30",
        "peft>=0.11",
    ])


def setup_repo():
    repo_url = os.environ.get(
        "NSA_REPO_URL",
        "https://github.com/aaliyan1230/neuron-suppression-awareness.git",
    )
    branch = os.environ.get("NSA_BRANCH", "main")
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", branch, repo_url, "/tmp/nsa",
    ])
    os.chdir("/tmp/nsa")
    sys.path.insert(0, "/tmp/nsa/src")


def setup_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    if not token:
        token_file = "/kaggle/working/.hf_token"
        if os.path.exists(token_file):
            with open(token_file) as f:
                token = f.read().strip()
    if not token:
        for token_file in (
            "/kaggle/input/nsa-hf-token/hf_token.txt",
            "/kaggle/input/nsa-hf-token/token.txt",
            "/kaggle/input/nsa-secrets/hf_token.txt",
            "/kaggle/input/nsa-secrets/token.txt",
        ):
            if os.path.exists(token_file):
                with open(token_file) as f:
                    token = f.read().strip()
                break
    if not token:
        for root, _dirs, files in os.walk("/kaggle/input"):
            for filename in files:
                if filename in {"hf_token.txt", "token.txt"}:
                    token_file = os.path.join(root, filename)
                    with open(token_file) as f:
                        token = f.read().strip()
                    print(f"Found HF token file at {token_file}")
                    break
            if token:
                break
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("HF token configured from Kaggle runtime inputs.")
    else:
        print("WARNING: No HF_TOKEN found. Model download may fail for gated models.")
        if os.path.exists("/kaggle/input"):
            print("Available /kaggle/input entries:", sorted(os.listdir("/kaggle/input")))


def run_phase5():
    from neuron_suppression_awareness.cli import main

    config_path = prepare_phase5_config()
    return main(["--config", config_path, "--backend", "transformers"])


def prepare_phase5_config():
    import yaml

    source = "/tmp/nsa/configs/phase5.qwen3_8b.kaggle_t4.yaml"
    phase2a_artifact_dir = resolve_phase2a_artifact_dir()
    with open(source, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["inputs"]["phase2a_artifact_dir"] = phase2a_artifact_dir
    output = "/kaggle/working/phase5_config.yaml"
    with open(output, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    print(f"Phase 2A artifacts resolved to {phase2a_artifact_dir}")
    print(f"Phase 5 runtime config written to {output}")
    return output


def resolve_phase2a_artifact_dir():
    local_snapshot = "/tmp/nsa/docs/phase2a-kaggle-run/artifacts/phase2a/20260527T184227Z"
    if has_phase2a_files(local_snapshot):
        return local_snapshot
    for root, _dirs, files in os.walk("/kaggle/input"):
        if PHASE2A_REQUIRED_FILES.issubset(set(files)):
            return root
    raise RuntimeError(
        "Could not find Phase 2A artifacts. Attach a Kaggle dataset containing "
        "caa_vectors.pt, concept_order.json, train_dataset.jsonl, and eval_dataset.jsonl."
    )


def has_phase2a_files(path):
    return os.path.isdir(path) and all(
        os.path.exists(os.path.join(path, filename))
        for filename in PHASE2A_REQUIRED_FILES
    )


if __name__ == "__main__":
    print("=== Installing dependencies ===")
    install_deps()

    print("=== Cloning repo ===")
    setup_repo()

    print("=== Setting up HF token ===")
    setup_hf_token()

    print("=== Running Phase 5: Mixed CAA + Suppression Detection Training ===")
    code = run_phase5()

    print(f"\n=== Phase 5 finished with exit code {code} ===")
    print("Artifacts written to /kaggle/working/artifacts/phase5/")
    sys.exit(code)
