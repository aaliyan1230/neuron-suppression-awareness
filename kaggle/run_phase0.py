"""
Kaggle kernel script for Phase 0 smoke test on T4 GPU.

Runs as a plain Python script via `kaggle kernels push`.
All output artifacts are written to /kaggle/working/ for retrieval.
"""

import os
import subprocess
import sys


def install_deps():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "transformers>=4.51", "datasets>=2.19",
        "huggingface-hub>=0.23", "pyyaml>=6.0",
        "bitsandbytes>=0.43", "accelerate>=0.30",
    ])


def setup_repo():
    repo_url = os.environ.get(
        "NSA_REPO_URL",
        "https://github.com/aaliyan1230/neuron-suppression-awareness.git",
    )
    branch = os.environ.get("NSA_BRANCH", "main")
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", branch, repo_url, "/tmp/nsa"])
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
        print("WARNING: No HF_TOKEN found. AdvBench (gated) may fail to load.")
        print("Options: attach a private Kaggle dataset with hf_token.txt or provide /kaggle/working/.hf_token.")
        if os.path.exists("/kaggle/input"):
            print("Available /kaggle/input entries:", sorted(os.listdir("/kaggle/input")))


def run_phase0():
    from neuron_suppression_awareness.cli import main
    config_path = "/tmp/nsa/configs/phase0.qwen3_8b.kaggle_t4.yaml"
    exit_code = main(["--config", config_path, "--backend", "transformers"])
    return exit_code


if __name__ == "__main__":
    print("=== Installing dependencies ===")
    install_deps()

    print("=== Cloning repo ===")
    setup_repo()

    print("=== Setting up HF token ===")
    setup_hf_token()

    print("=== Running Phase 0 smoke test ===")
    code = run_phase0()

    print(f"\n=== Phase 0 finished with exit code {code} ===")
    print("Artifacts written to /kaggle/working/artifacts/phase0/")
    sys.exit(code)
