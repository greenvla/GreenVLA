"""Build the infrastructure image and serve raw VLA from a compatible local checkpoint."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
PROFILE = json.loads(Path(__file__).with_name('profile.json').read_text())
PROFILES = {
    '014814': json.loads(Path(__file__).with_name('profile_014814.json').read_text()),
    '017799': {**PROFILE, 'data_config': 'dtwin_017799',
               'embodiment': 'dtwin_v102_filtered_s0s1_step1_state_action_episode_task'},
}
VLA_COMMIT = PROFILE['inference_baseline']


def checked_argv(args):
    """Allow only launcher tools; resolve them using the operator's trusted PATH.

    Arguments stay separate and literal. This is not a sandbox for Docker or
    Python: the local operator, repositories and inherited environment are trusted.
    """
    if not isinstance(args, (list, tuple)) or not args:
        raise RuntimeError('Command must be a non-empty argument list')
    argv = [str(x) for x in args]
    if argv[0] == sys.executable:
        # Preserve the virtualenv interpreter path, including any symlinks.
        return argv
    if argv[0] not in ('docker', 'git', 'nvidia-smi'):
        raise RuntimeError(f'Executable is not allowed by this launcher: {argv[0]}')
    executable = shutil.which(argv[0])
    if executable is None or not os.path.isabs(executable):
        raise RuntimeError(f'Missing system prerequisite or non-absolute PATH entry: {argv[0]}')
    return [executable, *argv[1:]]


def capture(*args):
    return subprocess.check_output(checked_argv(args), shell=False, text=True).strip()


def run(args, *, input=None, text=False, stdout=None, stderr=None):
    # No arbitrary Popen kwargs: callers cannot override shell or executable.
    subprocess.run(checked_argv(args), shell=False, check=True,
                   input=input, text=text, stdout=stdout, stderr=stderr)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_files(root, files):
    """Check layout only; historical profile digests are not enforced."""
    for rel in files:
        if not (root / rel).is_file():
            raise RuntimeError(f'Missing required artifact: {root / rel}')


def verify_hf(cache):
    spec = PROFILE['hf']
    repo = cache / ('models--' + spec['repo'].replace('/', '--'))
    require_files(repo / 'snapshots' / spec['revision'], spec['files'])
    ref = repo / 'refs/main'
    if not ref.is_file() or ref.read_text().strip() != spec['revision']:
        raise RuntimeError(f'HF refs/main must select the pinned revision; cache preserved: {ref}')


def select_gpu(text, requested, minimum):
    available = [tuple(int(x.strip()) for x in line.split(',')) for line in text.splitlines() if line.strip()]
    suitable = [(free, index) for index, free in available if free >= minimum and (requested is None or index == requested)]
    if not suitable:
        raise RuntimeError(f'Need {minimum} MiB free on one GPU; available {available}. No processes were stopped.')
    return max(suitable)[1]


def free_ports():
    for raw in range(9160, 9260, 2):
        sockets = [socket.socket(), socket.socket()]
        try:
            for s, port in zip(sockets, (raw, raw + 1)):
                s.bind(('0.0.0.0', port))
            return raw, raw + 1
        except OSError:
            continue
        finally:
            for s in sockets:
                s.close()
    raise RuntimeError('No free policy/tracking port pair in 9160..9259.')


def parser():
    p = argparse.ArgumentParser(prog='run_inference.sh', description=__doc__)
    p.add_argument('--prepare-only', action='store_true', help='Validate local checkpoint, build and prepare Qwen without starting GPU processes.')
    p.add_argument('--cache-dir', type=Path, default=REPO / '.best', help='Isolated cache; default .best/ in this checkout.')
    p.add_argument('--gpu', type=int, help='Default: GPU with most free memory; never stops other processes.')
    p.add_argument('--checkpoint', type=Path, required=True, help='Local checkpoint containing pretrained_model/ and norm_stats/. Never downloaded.')
    p.add_argument('--profile', choices=tuple(PROFILES), default='014814', help='Default: 014814 scripted embodiment. Use 017799 explicitly for the earlier checkpoint.')
    p.add_argument('--adapter', choices=('none', 'v1'), default='none', help='Optional learned joint correction; default: none.')
    p.add_argument('--neck-compensation', action='store_true', help='Opt in to the empirical neck bias; disabled by default.')
    p.add_argument('--hf-cache', type=Path, help='Optional existing pinned HF cache; default: use Qwen included in the image.')
    p.add_argument('--output', type=Path, help='New output directory; default .best/runs/<unique-name>.')
    p.add_argument('--dry-run', action='store_true', help='Print plan without building, downloading or writing files.')
    return p


def image_build(dockerfile, prefix, cache):
    # Hashes here name the Docker build cache, not validate runtime/checkpoint files.
    inputs = [REPO / dockerfile, REPO / 'pyproject.toml', REPO / 'docker/entrypoint.py',
              Path(__file__).with_name('profile.json'), Path(__file__).with_name('profile_014814.json'),
              Path(__file__).with_name('prepare_assets.py')]
    digest = hashlib.sha256(''.join(sha(f) for f in inputs).encode()).hexdigest()[:16]
    tag = f'{prefix}:{digest}'
    print(f'Building {tag}; log: {cache / (prefix + ".log")}', flush=True)
    with (cache / (prefix + '.log')).open('w') as log:
        run(['docker', 'build', '--progress', 'plain', '-f', REPO / dockerfile, '-t', tag, REPO], stdout=log, stderr=subprocess.STDOUT)
    return capture('docker', 'image', 'inspect', '-f', '{{.Id}}', tag)


def wait_healthy(port, name, seconds=360):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if capture('docker', 'inspect', '-f', '{{.State.Running}}', name) != 'true':
            raise RuntimeError(f'{name} exited; inspect docker logs. Container preserved.')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        threading.Event().wait(2)
    raise RuntimeError(f'{name} did not become healthy; inspect docker logs. Container preserved.')


def start_policy(image, checkpoint, hf_cache, output, tag, gpu, raw, tracking,
                 data_config='scripted_014814', adapter='none', neck_compensation=False):
    for part in ('policy', 'tracking'):
        (output / part).mkdir()
    common = ['docker', 'run', '-d', '--network', 'host', '--user', f'{os.getuid()}:{os.getgid()}',
        '--entrypoint', 'python', '-w', '/app', '-e', 'PYTHONPATH=/app', '-e', 'PYTHONDONTWRITEBYTECODE=1',
        '-e', 'TRITON_CACHE_DIR=/tmp/triton-cache', '-e', 'XDG_CACHE_HOME=/tmp/runtime-cache',
        '-e', 'TORCHINDUCTOR_CACHE_DIR=/tmp/torch-cache', '-e', 'USER=runtime', '-e', 'HOME=/tmp/runtime-home',
        '-v', f'{REPO}:/app:ro', '-v', f'{output}:/output']
    hf_mount = ['-v', f'{hf_cache}:/opt/hf-cache:ro'] if hf_cache is not None else []
    run(common + ['--name', tag + '_policy', '--gpus', f'device={gpu}',
        '-e', f'POLICY_PORT={raw}',
        '-e', 'HF_HOME=/tmp/hf-home', '-e', 'HF_HUB_CACHE=/opt/hf-cache',
        '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'S0S1_PROMPT_FROM_SUBTASK=1',
        '-e', 'S0S1_SWAP_WRISTS=1', '-e', 'S0S1_GMM_GUARD_MODE=off', '-e', 'CONTRACT_AUDIT_DIR=/output/policy',
        '-v', f'{checkpoint}:/ckpt:ro'] + hf_mount + [image,
        '-m', 'lerobot.scripts.obm_inference.serve_policy', '--checkpoint', '/ckpt', '--data-config', data_config,
        '--device', 'cuda:0', '--port', raw, '--feed-by-server', '--feed.row-dt', '0.03333333333333333', '--warmup-steps', '2'])
    wait_healthy(raw, tag + '_policy')
    run(common + ['--name', tag + '_tracking', '-e', f'POLICY_PORT={tracking}',
        '-e', 'OPENBLAS_NUM_THREADS=2', '-e', 'OMP_NUM_THREADS=2', image,
        '-m', 'lerobot.scripts.obm_inference.serve_tracking', '--upstream-port', raw, '--port', tracking,
        '--adapter', adapter, '--log', '/output/tracking/boundary.jsonl'] +
        (['--neck-compensation'] if neck_compensation else []))
    wait_healthy(tracking, tag + '_tracking', 60)


def main():
    a = parser().parse_args()
    selected = PROFILES[a.profile]
    checkpoint_files = list(selected['checkpoint']['files'])
    cache = a.cache_dir.expanduser().resolve()
    checkpoint = a.checkpoint.expanduser().resolve()
    hf_cache = a.hf_cache.expanduser().resolve() if a.hf_cache else None
    print(f'Profile: {selected["name"]}; embodiment: {selected["embodiment"]}; cache: {cache}; checkpoint: {checkpoint}', flush=True)
    print(f'Runtime: adapter={a.adapter}, neck_compensation={a.neck_compensation}, GMM=off. '
          'Simulator contract: ARM_FF=0 PROPRIO_CMD=all (set these in an external simulator).', flush=True)
    if a.dry_run:
        print('Validate/mount the local checkpoint read-only; build docker/Dockerfile; start VLA + wire boundary.')
        print('Use the supplied HF cache without runtime downloads.' if a.hf_cache else
              'Use the pinned Qwen snapshot included in the image; runtime is offline.')
        return
    require_files(checkpoint, checkpoint_files)
    if a.hf_cache:
        verify_hf(hf_cache)
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('Requires Linux x86_64 with NVIDIA GPU and Docker/NVIDIA Container Toolkit.')
    if not shutil.which('docker'):
        raise RuntimeError('Missing system prerequisite: docker')
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another launch is active for {cache}; no duplicate run started.')
        policy_image = image_build('docker/Dockerfile', 'green-vla-best', cache)
        require_files(checkpoint, checkpoint_files)
        if hf_cache is not None:
            verify_hf(hf_cache)
        if a.prepare_only:
            print(f'Prepared. Policy image: {policy_image}. No GPU process started.', flush=True)
            return
        gpu = select_gpu(capture('nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'), a.gpu, 7000)
        raw, tracking = free_ports()
        tag = 'q35_best_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
        output = (a.output or cache / 'runs' / tag).resolve()
        if output.exists():
            raise RuntimeError(f'Output already exists; preserved: {output}')
        output.mkdir(parents=True, exist_ok=False)
        receipt = {'profile': selected['name'], 'tag': tag, 'inference_baseline': VLA_COMMIT,
            'checkpoint': str(checkpoint), 'checkpoint_files': checkpoint_files,
            'hf_revision': PROFILE['hf']['revision'], 'policy_image': policy_image,
            'gpu': gpu, 'raw_port': raw, 'policy_port': tracking, 'output': str(output),
            'external_simulator_requirements': {'ARM_FF': '0', 'PROPRIO_CMD': 'all'},
            'contract': {'data_config': selected['data_config'], 'embodiment': selected['embodiment'],
                         'adapter': a.adapter, 'neck_compensation': a.neck_compensation,
                         'gmm': 'off', 'proprio_cmd': 'all',
                         'native_subtasks': True, 'robot': 'green', 'flow_steps': 10, 'horizon': 50, 'source_dt': 1/30}}
        (output / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(f'Launching checkpoint {a.profile}; adapter={a.adapter}, neck_compensation={a.neck_compensation}; '
              f'GPU {gpu}, ports {raw}/{tracking}; {output}', flush=True)
        start_policy(policy_image, checkpoint, hf_cache, output, tag, gpu, raw, tracking,
                     selected['data_config'], a.adapter, a.neck_compensation)
        print(f'VLA ready: ws://127.0.0.1:{tracking} (tracking endpoint).', flush=True)
        print(f'Complete. Servers retained: {tag}_policy / {tag}_tracking; artifacts: {output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Launch failed: {exc}', file=sys.stderr)
        sys.exit(1)
