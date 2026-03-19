import subprocess
from pathlib import Path

import modal

APP_NAME = "deepseekv3-moe-c"
PROJECT_DIR = "/root/project"

app = modal.App(APP_NAME)

# image = (
#     modal.Image.debian_slim(python_version="3.11")
#     .apt_install("build-essential", "git")
#     .pip_install(
#         "torch",
#         "numpy",
#         "accelerate",
#         "safetensors",
#         "sentencepiece",
#         "einops",
#         "git+https://github.com/huggingface/transformers.git",
#     )
#     .add_local_dir(".", remote_path=PROJECT_DIR)
# )

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .pip_install("numpy")
    .add_local_dir(".", remote_path=PROJECT_DIR)
)

@app.function(
    image=image,
    cpu=4,
    timeout=60 * 60,
)
def run_all():
    project = Path(PROJECT_DIR)

    def run(cmd, cwd=project):
        print(f"\n>>> RUNNING: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        return proc.stdout

    run(["python", "export_testcases.py"])
    run(["make", "clean"])
    run(["make"])
    test_output = run(["./test_moe"])

    case_dir = project / "testdata" / "case_001"
    files = []
    if case_dir.exists():
        files = sorted(str(p.relative_to(project)) for p in case_dir.rglob("*") if p.is_file())

    return {
        "status": "ok",
        "files": files,
        "test_output": test_output,
    }


@app.local_entrypoint()
def main():
    result = run_all.remote()
    print("\n=== RESULT ===")
    print(result["status"])
    print("\nGenerated files:")
    for f in result["files"]:
        print(f"  - {f}")
    print("\n=== TEST OUTPUT ===")
    print(result["test_output"])