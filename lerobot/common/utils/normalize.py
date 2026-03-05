import json
import pathlib

import numpy as np
import numpydantic
import pydantic
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader

from lerobot.common.utils.seed_utils import _seed_worker

@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): A 2D array where each row is a new vector.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [
                np.zeros(self._num_quantile_bins) for _ in range(vector_length)
            ]
            self._bin_edges = [
                np.linspace(
                    self._min[i] - 1e-10,
                    self._max[i] + 1e-10,
                    self._num_quantile_bins + 1,
                )
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError(
                    "The length of new vectors does not match the initialized vector length."
                )
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(
                self._min[i], self._max[i], self._num_quantile_bins + 1
            )

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(
                old_edges[:-1], bins=new_edges, weights=self._histograms[i]
            )

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results

    def merge(self, other: "RunningStats") -> None:
        """
        Merge another RunningStats instance (`other`) into this one (`self`),
        so that `self` afterwards reflects the statistics over all data seen
        by both. Modifies `self` in place.
        """
        # If one side is empty, just copy the other over
        if self._count == 0:
            # Shallow‐copy everything from other
            self._count = other._count
            self._mean = None if other._mean is None else other._mean.copy()
            self._mean_of_squares = (
                None
                if other._mean_of_squares is None
                else other._mean_of_squares.copy()
            )
            self._min = None if other._min is None else other._min.copy()
            self._max = None if other._max is None else other._max.copy()
            # Deep‐copy lists of histograms and bin_edges
            if other._histograms is not None:
                self._histograms = [h.copy() for h in other._histograms]
                self._bin_edges = [e.copy() for e in other._bin_edges]
            else:
                self._histograms = None
                self._bin_edges = None
            return
        if other._count == 0:
            # Nothing to do; `self` already has data
            return

        # Ensure the two stats cover the same vector dimensionality
        dim = self._mean.size
        if other._mean.size != dim:
            raise ValueError("Cannot merge RunningStats: dimension mismatch.")

        # 1) Combine counts, means, means of squares
        n1 = self._count
        n2 = other._count
        total_n = n1 + n2

        # new mean = (n1*m1 + n2*m2) / (n1+n2)
        m1 = self._mean
        m2 = other._mean
        new_mean = (n1 * m1 + n2 * m2) / total_n

        # new mean_of_squares = (n1*s1 + n2*s2) / (n1+n2)
        s1 = self._mean_of_squares
        s2 = other._mean_of_squares
        new_mean_of_squares = (n1 * s1 + n2 * s2) / total_n

        # 2) Combine min/max
        new_min = np.minimum(self._min, other._min)
        new_max = np.maximum(self._max, other._max)

        # 3) Build new bin edges for each dimension, spanning [new_min[i], new_max[i]]
        num_bins = self._num_quantile_bins
        new_bin_edges = [
            np.linspace(new_min[i], new_max[i], num_bins + 1) for i in range(dim)
        ]

        # 4) Re‐bin both sides' histograms onto the new edges, then sum
        def _rebin_histogram(
            old_hist: np.ndarray, old_edges: np.ndarray, new_edges: np.ndarray
        ) -> np.ndarray:
            """
            Given an existing histogram `old_hist` with bin‐edges `old_edges`,
            produce a new histogram (same total counts) on `new_edges` by
            treating each old bin as a point at its midpoint with weight = count.
            """
            # Compute midpoint of each old bin
            old_midpoints = (old_edges[:-1] + old_edges[1:]) / 2.0
            # Re‐distribute counts to the new bins by weighting the midpoints
            rebinned, _ = np.histogram(old_midpoints, bins=new_edges, weights=old_hist)
            return rebinned

        new_histograms: list[np.ndarray] = []
        for i in range(dim):
            # old data from self
            hist1 = self._histograms[i]
            edges1 = self._bin_edges[i]
            rebinned1 = _rebin_histogram(hist1, edges1, new_bin_edges[i])

            # old data from other
            hist2 = other._histograms[i]
            edges2 = other._bin_edges[i]
            rebinned2 = _rebin_histogram(hist2, edges2, new_bin_edges[i])

            # Sum the two re‐binned histograms
            new_histograms.append(rebinned1 + rebinned2)

        # 5) Assign back into self
        self._count = total_n
        self._mean = new_mean
        self._mean_of_squares = new_mean_of_squares
        self._min = new_min
        self._max = new_max
        self._bin_edges = new_bin_edges
        self._histograms = new_histograms


def compute_dataset_stats_from_dataset(dataset, max_samples=100000):
    dataset_name = dataset.get_dataset_summary()["dataset_id"]
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16, worker_init_fn=_seed_worker)

    return_fake_images = dataloader.dataset._dataset._dataset.return_fake_images
    dataloader.dataset._dataset._dataset.return_fake_images = True

    max_steps = None
    if max_samples is not None:
        max_steps = (
            max_samples
            // dataloader.batch_size
        ) + 1

    keys = ["state", "actions"]
    stats = {key: RunningStats() for key in keys}


    if max_steps is not None:
        max_steps = min(max_steps, len(dataloader))
    else:
        max_steps = len(dataloader)

    total_steps = 0
    for batch in tqdm(
        dataloader,
        total=max_steps,
        desc=f"Computing dataset stats for {dataset_name}",
    ):
        for key in keys:
            values = batch[key].detach().cpu().numpy()
            stats[key].update(values.reshape(-1, values.shape[-1]))

        total_steps += 1
        if max_steps is not None and total_steps >= max_steps:
            break

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_stats = [None] * world_size
        dist.all_gather_object(gathered_stats, stats)
    else:
        gathered_stats = [stats]

    for i in range(1, len(gathered_stats)):
        for key in keys:
            gathered_stats[0][key].merge(gathered_stats[i][key])

    norm_stats = {
        key: stats.get_statistics() for key, stats in gathered_stats[0].items()
    }
    dataloader.dataset._dataset._dataset.return_fake_images = return_fake_images
    return norm_stats

class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def norm_stats_to_dict(norm_stats: dict[str, NormStats]) -> dict:
    """Serialize the running statistics to a dictionary."""
    return {
        k: {
            "mean": v.mean.tolist(),
            "std": v.std.tolist(),
            "q01": v.q01.tolist(),
            "q99": v.q99.tolist(),
        }
        for k, v in norm_stats.items()
    }


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
