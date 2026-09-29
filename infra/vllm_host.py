"""Self-hosted vLLM on RunPod or AWS via SkyPilot, reached through an SSH tunnel.

    uv run --group infra python infra/vllm_host.py up --config deploy/qwen35_9b.yaml --gpus H100:1 --backend runpod
    uv run --group infra python infra/vllm_host.py status
    uv run --group infra python infra/vllm_host.py tunnel   # reopen the tunnel if it dropped
    uv run --group infra python infra/vllm_host.py down

A deployment file pins the vLLM version and holds the engine settings for it (`engine:` +
`vllm:`, see deploy/qwen35_9b.yaml); it knows nothing about hardware. vLLM is pip-installed
into its own venv on the node, so RunPod and AWS run the same engine. `--gpus` and
`--backend` pick the machine; the tensor-parallel size is derived from the GPU count on the
node.

The server listens on 127.0.0.1 on the node only. `up` opens `ssh -L` from this machine and
writes `.vllm_endpoint.json` (gitignored) with the local base_url and a per-deployment API
key. The cluster is deleted after `--max-minutes` no matter what, so a forgotten server
doesn't keep billing.
"""

import json
import os
import re
import secrets
import signal
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import sky
import typer
import yaml
from sky.skylet.autostop_lib import AutostopWaitFor

REMOTE_CONFIG = "/config/vllm.yaml"
REMOTE_PORT = 8000
ENDPOINT_FILE = Path(".vllm_endpoint.json")

app = typer.Typer(add_completion=False, no_args_is_help=True)


def load_deployment(path: Path) -> tuple[str, str | None, dict[str, str], Path]:
    """Split a deployment file into its pinned engine, CUDA build, engine env vars, and a vLLM config to mount."""
    deployment = yaml.safe_load(path.read_text())
    if not {"engine", "vllm"} <= set(deployment) <= {"engine", "cuda", "env", "vllm"}:
        raise typer.BadParameter(f"{path} must have `engine:` and `vllm:` keys, and optionally `cuda:` and `env:`")
    if not re.fullmatch(r"vllm==[\w.+-]+", deployment["engine"]):
        raise typer.BadParameter("engine must be an exact pin, e.g. vllm==0.30.0")
    cuda = deployment.get("cuda")
    if cuda is not None and not re.fullmatch(r"cu\d{3}", cuda):
        raise typer.BadParameter("cuda must look like cu129")
    if "tensor-parallel-size" in deployment["vllm"]:
        raise typer.BadParameter("tensor-parallel-size comes from --gpus, not the deployment file")
    vllm_config = Path(tempfile.mkdtemp()) / "vllm.yaml"
    vllm_config.write_text(yaml.safe_dump(deployment["vllm"], sort_keys=False))
    env = {str(k): str(v) for k, v in (deployment.get("env") or {}).items()}
    return deployment["engine"], cuda, env, vllm_config


def install_spec(engine: str, cuda: str | None) -> tuple[str, str]:
    """(uv pip install arguments, venv suffix) for the pinned engine.

    With `cuda`, install vLLM's CUDA-specific wheel from its GitHub release together with the
    matching PyTorch build. The PyPI wheel targets the newest CUDA, which needs a newer driver
    than many RunPod hosts have; a CUDA 12.x build runs on those and on newer drivers alike.
    """
    if cuda is None:
        return f"'{engine}'", engine.replace("==", "-")
    version = engine.removeprefix("vllm==")
    wheel = (
        f"https://github.com/vllm-project/vllm/releases/download/v{version}/"
        f"vllm-{version}%2B{cuda}-cp38-abi3-manylinux_2_28_x86_64.whl"
    )
    return f"--torch-backend={cuda} 'vllm @ {wheel}'", f"vllm-{version}-{cuda}"


OFFLOAD_RESERVE_GIB = 12  # left for the OS, the vLLM/Python processes and page cache


def cpu_offload_snippet(cpu_offload: str) -> str:
    """Shell that sets $OFFLOAD_ARGS for `vllm serve` (KV-cache offloading to CPU memory).

    "0": off. "<N>": N GiB. "auto": the node's available memory, capped by the container's cgroup
    limit (RunPod's /proc/meminfo can show the whole host), minus OFFLOAD_RESERVE_GIB. Sized on the
    node because it depends on the machine, like --tensor-parallel-size.
    """
    if cpu_offload == "0":
        return 'OFFLOAD_ARGS=""'
    if cpu_offload != "auto":
        return f'OFFLOAD_ARGS="--kv-offloading-size {int(cpu_offload)}"'
    return (
        'mem_kb=$(awk \'/MemAvailable/ {print $2}\' /proc/meminfo); '
        'lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max); '
        'case "$lim" in max|"") ;; *) lim_kb=$((lim / 1024)); [ "$lim_kb" -lt "$mem_kb" ] && mem_kb=$lim_kb ;; esac; '
        f"gib=$(( mem_kb / 1048576 - {OFFLOAD_RESERVE_GIB} )); "
        'if [ "$gib" -gt 0 ]; then OFFLOAD_ARGS="--kv-offloading-size $gib"; else OFFLOAD_ARGS=""; fi; '
        'echo "CPU KV offload: ${gib} GiB"'
    )


def build_task(
    deployment: Path, gpus: str, backend: str, api_key: str, disk_size: int, cpu_offload: str = "0"
) -> sky.Task:
    engine, cuda, env, vllm_config = load_deployment(deployment)
    install_args, venv_suffix = install_spec(engine, cuda)
    # One venv per engine build, with its own Python, independent of the base image.
    venv = f"~/venv-{venv_suffix}"
    setup = (
        "set -e; export PATH=$HOME/.local/bin:$PATH; "
        f"if [ ! -x {venv}/bin/vllm ]; then "
        "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; "
        # --clear: an interrupted install leaves a venv without `vllm`; start it over.
        f"uv venv --clear {venv} --python 3.12; "
        f"uv pip install --python {venv}/bin/python {install_args}; "
        "fi"
    )
    run = (
        f"{cpu_offload_snippet(cpu_offload)}; "
        f"{venv}/bin/vllm serve --config {REMOTE_CONFIG} --host 127.0.0.1 --port {REMOTE_PORT} "
        "--tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE $OFFLOAD_ARGS"
    )
    task = sky.Task(
        name="vllm",
        setup=setup,
        run=run,
        envs=env,
        file_mounts={REMOTE_CONFIG: str(vllm_config)},
        secrets={"VLLM_API_KEY": api_key},  # vLLM reads it; SkyPilot masks secrets in logs
    )
    # "A40:1,L40S:1" means any of these: SkyPilot takes the cheapest one with capacity.
    task.set_resources(
        {
            sky.Resources(infra=backend, accelerators=spec.strip(), disk_size=disk_size)
            for spec in gpus.split(",")
        }
    )
    return task


def open_tunnel(cluster: str, local_port: int) -> int:
    """ssh -N -L via SkyPilot's generated SSH config (the cluster name is the host alias)."""
    proc = subprocess.Popen(
        ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
         "-L", f"{local_port}:127.0.0.1:{REMOTE_PORT}", cluster],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,  # survives this script exiting
    )
    time.sleep(3)
    if proc.poll() is not None:
        raise typer.Exit(f"SSH tunnel to {cluster} exited immediately (code {proc.returncode})")
    return proc.pid


def close_tunnel(pid: int | None) -> None:
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def get(url: str, api_key: str | None = None, timeout: float = 5) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except Exception as e:  # not up yet, tunnel down, auth error...
        return getattr(e, "code", 0), str(e)


def wait_until_ready(base_url: str, api_key: str, timeout_s: int) -> str:
    """Poll through the tunnel until the model is listed; returns the served model id."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, body = get(f"{base_url}/models", api_key)
        if status == 200:
            return json.loads(body)["data"][0]["id"]
        time.sleep(10)
    raise typer.Exit(f"vLLM not ready after {timeout_s}s; check `sky logs <cluster>`")


def read_endpoint() -> dict:
    if not ENDPOINT_FILE.exists():
        raise typer.Exit(f"No {ENDPOINT_FILE}; run `up` first")
    return json.loads(ENDPOINT_FILE.read_text())


def write_endpoint(endpoint: dict) -> None:
    ENDPOINT_FILE.write_text(json.dumps(endpoint, indent=2) + "\n")
    ENDPOINT_FILE.chmod(0o600)


@app.command()
def up(
    config: Path = typer.Option(..., help="Deployment file: pinned vLLM version + engine settings"),
    gpus: str = typer.Option(
        "H100:1", help="SkyPilot accelerator spec, or a comma-separated any-of list, e.g. A40:1,RTXA6000:1,L40S:1"
    ),
    backend: str = typer.Option(..., help="runpod or aws"),
    cluster: str = typer.Option("sd-vllm", help="SkyPilot cluster name (also the SSH host alias)"),
    local_port: int = typer.Option(8000, help="Local end of the SSH tunnel"),
    max_minutes: int = typer.Option(240, help="Delete the cluster after this long, unconditionally"),
    disk_size: int = typer.Option(100, help="Disk GB (image + weights)"),
    ready_timeout: int = typer.Option(1800, help="Seconds to wait for the model to load"),
    cpu_offload: str = typer.Option(
        "0", help="KV-cache offload to CPU memory: GiB, 'auto' (node memory minus a reserve), or 0 = off (untested)"
    ),
) -> None:
    """Launch vLLM, open the tunnel, wait until it serves, write .vllm_endpoint.json."""
    if backend not in ("runpod", "aws"):
        raise typer.BadParameter("--backend must be runpod or aws")
    if cpu_offload != "auto" and not (cpu_offload.isdigit()):
        raise typer.BadParameter("--cpu-offload must be a number of GiB, 'auto' or 0")
    api_key = secrets.token_urlsafe(32)
    task = build_task(config, gpus, backend, api_key, disk_size, cpu_offload)

    existed = bool(sky.get(sky.status([cluster])))
    typer.echo(f"Launching {cluster} on {backend} with {gpus} (deleted after {max_minutes} min)...")
    try:
        sky.stream_and_get(
            sky.launch(
                task,
                cluster_name=cluster,
                idle_minutes_to_autostop=max_minutes,
                wait_for=AutostopWaitFor.NONE,  # a serving job never goes idle; this is a hard lifetime cap
                down=True,
            )
        )
    except BaseException:
        if existed:
            # Relaunching onto a cluster that was already up: leave it for the user to inspect.
            typer.echo(f"Launch failed on existing cluster {cluster}; it's still up (`sky down {cluster}` to delete)")
            raise
        # A fresh launch can fail after the node exists (e.g. SSH never comes up), before the
        # lifetime cap is armed: delete it rather than leave it billing.
        typer.echo(f"Launch failed; deleting {cluster} if it was created...")
        try:
            sky.stream_and_get(sky.down(cluster))
        except Exception as e:
            typer.echo(f"Cleanup failed ({e}); check `sky status` and run `sky down {cluster}`")
        raise

    pid = open_tunnel(cluster, local_port)
    base_url = f"http://127.0.0.1:{local_port}/v1"
    endpoint = {"cluster": cluster, "backend": backend, "gpus": gpus, "config": str(config),
                "base_url": base_url, "api_key": api_key, "tunnel_pid": pid, "local_port": local_port}
    write_endpoint(endpoint)  # written before waiting, so `down` can clean up if loading fails

    typer.echo("Waiting for the model to load (weights download on first start)...")
    model = wait_until_ready(base_url, api_key, ready_timeout)
    write_endpoint({**endpoint, "model": model})
    typer.echo(f"Ready: {model} at {base_url} (API key in {ENDPOINT_FILE})")


@app.command()
def tunnel() -> None:
    """Reopen the SSH tunnel for the current deployment."""
    endpoint = read_endpoint()
    close_tunnel(endpoint.get("tunnel_pid"))
    endpoint["tunnel_pid"] = open_tunnel(endpoint["cluster"], endpoint["local_port"])
    write_endpoint(endpoint)
    typer.echo(f"Tunnel open: {endpoint['base_url']}")


@app.command()
def status() -> None:
    """Cluster state from SkyPilot plus a health check through the tunnel."""
    endpoint = read_endpoint()
    for record in sky.get(sky.status([endpoint["cluster"]], refresh=sky.StatusRefreshMode.AUTO)):
        typer.echo(f"{record['name']}: {record['status']} ({endpoint['backend']}, {endpoint['gpus']})")
    code, body = get(f"{endpoint['base_url']}/models", endpoint["api_key"])
    typer.echo(f"API: {'up' if code == 200 else f'unreachable ({body})'} at {endpoint['base_url']}")


@app.command()
def down() -> None:
    """Close the tunnel and delete the cluster."""
    endpoint = read_endpoint()
    close_tunnel(endpoint.get("tunnel_pid"))
    sky.stream_and_get(sky.down(endpoint["cluster"]))
    ENDPOINT_FILE.unlink()
    typer.echo(f"Deleted {endpoint['cluster']}")


if __name__ == "__main__":
    app()
