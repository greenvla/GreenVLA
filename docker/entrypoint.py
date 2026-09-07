"""Infrastructure entrypoint: raw VLA + required simulator wire conversions."""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ASSETS = Path(__file__).resolve().parents[1] / 'lerobot/common/robot_safety/tracking/assets'


def commands(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT_PATH', '/ckpt'))
    parser.add_argument('--data-config', default=os.environ.get('S0S1_DATA_CONFIG', 'scripted_014814'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('POLICY_PORT', '8999')))
    parser.add_argument('--raw-port', type=int, default=int(os.environ.get('POLICY_RAW_PORT', '8998')))
    parser.add_argument('--adapter', choices=('none', 'v1'), default='none')
    parser.add_argument('--neck-compensation', action='store_true')
    args, extra = parser.parse_known_args(argv)
    if not (0 < args.port < 65536 and 0 < args.raw_port < 65536) or args.port == args.raw_port:
        parser.error('Policy and internal raw ports must be valid and different')
    if args.port != int(os.environ.get('POLICY_PORT', '8999')):
        parser.error('Set POLICY_PORT to the same public port so the Docker healthcheck follows it')
    raw = [sys.executable, '-m', 'lerobot.scripts.obm_inference.serve_policy',
           '--checkpoint', args.checkpoint, '--data-config', args.data_config,
           '--host', '127.0.0.1', '--port', str(args.raw_port), '--device', 'cuda:0',
           '--feed-by-server', '--feed.row-dt', '0.03333333333333333', '--warmup-steps', '2', *extra]
    tracking = [sys.executable, '-m', 'lerobot.scripts.obm_inference.serve_tracking',
                '--upstream-port', str(args.raw_port), '--port', str(args.port),
                '--joint-calibration', str(ASSETS / 'joint_tracking_calibration_v1.json'),
                '--adapter', args.adapter,
                '--log', os.environ.get('TRACKING_LOG', '/tmp/green-vla/tracking.jsonl')]
    if args.neck_compensation:
        tracking.append('--neck-compensation')
    return args, raw, tracking


def wait_ready(process, port, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Policy exited before readiness: {process.returncode}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        threading.Event().wait(0.5)
    raise RuntimeError(f'Policy was not ready after {seconds}s')


def main(argv=None):
    args, raw, tracking = commands(argv)
    # Reject occupied ports instead of attaching the boundary to another policy.
    for port in (args.raw_port, args.port):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('0.0.0.0', port))
    env = os.environ.copy()
    for key, value in {
        'S0S1_PROMPT_FROM_SUBTASK': '1', 'S0S1_SWAP_WRISTS': '1', 'S0S1_GMM_GUARD_MODE': 'off',
        'TRITON_CACHE_DIR': '/tmp/triton-cache', 'XDG_CACHE_HOME': '/tmp/runtime-cache',
        'TORCHINDUCTOR_CACHE_DIR': '/tmp/torch-cache',
    }.items():
        env.setdefault(key, value)
    children = []

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        print(f'Starting {args.data_config}: adapter={args.adapter}, '
              f'neck_compensation={args.neck_compensation}, GMM={env["S0S1_GMM_GUARD_MODE"]}; '
              f'port {args.port}. Configure the simulator separately: ARM_FF=0 PROPRIO_CMD=all.', flush=True)
        # Keep argv separate: checkpoint paths and policy options are not shell code.
        children.append(subprocess.Popen(raw, shell=False, env=env))
        wait_ready(children[0], args.raw_port, float(env.get('POLICY_START_TIMEOUT', '360')))
        children.append(subprocess.Popen(
            tracking, shell=False, env={**env, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}))
        while all(child.poll() is None for child in children):
            threading.Event().wait(0.5)
        # The proxy holds a persistent upstream connection: restart the whole
        # container if either server exits, never leave a stale healthy proxy.
        return next((child.returncode or 1 for child in children if child.returncode is not None), 1)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as exc:
        print(f'Inference startup failed: {exc}', file=sys.stderr)
        sys.exit(1)
