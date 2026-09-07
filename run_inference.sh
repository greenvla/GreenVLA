#!/usr/bin/env bash
# Local checkpoint, raw VLA by default, with required simulator wire conversions.
set -euo pipefail
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tools/repro017799/launch_best.py" "$@"
