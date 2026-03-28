"""Tensor helpers for graph observation batches."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

GraphObservation = Dict[str, np.ndarray]
GraphTensorBatch = Dict[str, torch.Tensor]


def stack_graph_observations(observations: List[GraphObservation], device: torch.device) -> GraphTensorBatch:
    """Stack observation dictionaries into batched tensors."""
    return {
        key: torch.as_tensor(np.stack([obs[key] for obs in observations], axis=0), dtype=torch.float32, device=device)
        for key in observations[0]
    }


def single_graph_observation(observation: GraphObservation, device: torch.device) -> GraphTensorBatch:
    """Wrap a single observation into a batch of size one."""
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for key, value in observation.items()
    }


def batch_slice(batch: GraphTensorBatch, indices: torch.Tensor) -> GraphTensorBatch:
    """Index every tensor in a graph batch consistently."""
    return {key: value.index_select(0, indices) for key, value in batch.items()}
