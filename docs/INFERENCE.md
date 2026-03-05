# Inference with Green-VLA

> Guide for running inference with Green-VLA models.

## Table of Contents

- [Quick Inference](#quick-inference)
- [Examples](#examples)
- [Benchmarking Notes](#benchmarking-notes)

---

## Quick Inference

```python
from lerobot.common.policies.factory import load_pretrained_policy

# Load from HuggingFace Hub
policy, input_transforms, output_transforms = load_pretrained_policy(
    "SberRoboticsCenter/GreenVLA-5b-base",
    data_config_name="bridge",
)

# Or load from a local checkpoint
policy, input_transforms, output_transforms = load_pretrained_policy(
    "/path/to/checkpoints/300000",
    data_config_name="bridge",
    config_overrides={"device": "cuda:0"},
)
```

---

## Examples

For full end-to-end inference scripts with argument parsing, see the examples directory:

| Script | Description |
|--------|-------------|
| [`example_inference_bridge.py`](../examples/example_inference_bridge.py) | Action inference on Bridge (WidowX) |
| [`example_inference_fractal.py`](../examples/example_inference_fractal.py) | Action inference on Fractal (Google Robot) |
| [`example_vqa.ipynb`](../examples/example_vqa.ipynb) | VQA, bounding box detection, and pointing |

---

## Benchmarking Notes

- For **Bridge (WidowX)** benchmarking on SimplerEnv we used `action_horizon=2`.
- Bridge benchmark results can vary up to **±6%** between runs. We recommend averaging over multiple evaluation runs for reliable comparisons.
