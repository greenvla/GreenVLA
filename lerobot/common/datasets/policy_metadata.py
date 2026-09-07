"""The one piece of dataset metadata handling the policy prompt needs.

The rest of the upstream module builds metadata while a dataset is assembled,
which is training-side work and is not shipped here.
"""

from collections.abc import Mapping
from typing import Any


def format_policy_metadata_prompt(metadata: Mapping[str, Any]) -> str:
    negative = str(bool(metadata.get("negative", False))).lower()
    dropped = set(metadata.get("_dropped_fields", ()))
    fields = [
        ("task", f"Task: {metadata.get('task', '')}"),
        ("subtask", f"Subtask: {metadata.get('subtask', '')}"),
        ("phase", f"Phase: {metadata.get('phase', '')}"),
        ("mode", f"Mode: {metadata.get('mode', 'normal')}"),
        ("negative", f"Negative: {negative}"),
        ("error", f"Error: {metadata.get('error', 'none')}"),
        ("speed", f"Speed: {metadata.get('speed', 'normal')}"),
        ("hand", f"Hand: {metadata.get('hand', 'hightorque')}"),
        ("cameras", f"Cameras: {metadata.get('cameras', 'real')}"),
    ]
    if metadata.get("quality") is not None:
        fields.append(("quality", f"Quality: {metadata['quality']}"))
    return "\n".join(
        text
        for field, text in fields
        if field not in dropped
        and (field not in {"task", "subtask", "phase"} or metadata.get(field))
    )
