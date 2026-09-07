"""Prepare the pinned Qwen HF snapshot only. Checkpoint is always supplied locally."""
import argparse
import json
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

PROFILE = json.loads((Path(__file__).with_name('profile.json')).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    a = p.parse_args()
    # Credentials, when provided as environment values by the host, travel on
    # stdin. They are not Docker build args, command-line values, logs or files.
    secret = json.loads(sys.stdin.read() or '{}')
    spec = PROFILE['hf']
    model_cache = a.cache / 'hf-cache'
    repo_cache = model_cache / ('models--' + spec['repo'].replace('/', '--'))
    ref = repo_cache / 'refs/main'
    if ref.exists() and ref.read_text().strip() != spec['revision']:
        raise RuntimeError(f'Existing HF cache points at another revision: {ref}')
    snapshot = repo_cache / 'snapshots' / spec['revision']
    missing = [name for name in spec['files'] if not (snapshot / name).is_file()]
    if missing:
        print(f'Downloading {spec["repo"]}@{spec["revision"]}', flush=True)
        snapshot_download(spec['repo'], revision=spec['revision'], cache_dir=model_cache,
                          allow_patterns=missing, token=secret.get('HF_TOKEN'), max_workers=4)
    for name in spec['files']:
        if not (snapshot / name).is_file():
            raise RuntimeError(f'Missing pinned HF artifact: {name}')
    # Runtime resolves the model by repo ID; bind main to the pinned snapshot
    # in this isolated cache. No model/config/tokenizer content is changed.
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(spec['revision'])
    print('Artifact checks passed. No credentials were saved.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Do not dump request headers, signed URLs, credential dictionaries or env.
        print(f'Artifact preparation failed ({type(exc).__name__}): check source access and pinned files.', file=sys.stderr)
        if isinstance(exc, RuntimeError):
            print(str(exc), file=sys.stderr)
        sys.exit(1)
