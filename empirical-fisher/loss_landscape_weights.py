# Copyright (c) 2017 Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein
#
# Extracted from the “loss landscape” reference code (``net_plotter.py``) for use
# without h5py / HDF5.  See ``LICENSE.loss_landscape`` for the full MIT license text.

from __future__ import annotations

import torch

__all__ = [
    "create_random_direction",
    "get_weights",
    "set_weights",
]


def get_weights(net):
    """Extract parameters from net, and return a list of tensors."""
    return [p.data for p in net.parameters()]


def set_weights(net, weights, directions=None, step=None):
    """Overwrite the network's weights or move along ``directions`` by ``step``."""
    if directions is None:
        for (p, w) in zip(net.parameters(), weights, strict=False):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, (
            "If a direction is specified then step must be specified as well"
        )

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy, strict=False)]
        else:
            changes = [d * step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes, strict=False):
            p.data = w + torch.Tensor(d).type(type(w))


def get_random_weights(weights):
    """Random Gaussian tensors with the same shape as the network's weights."""
    return [torch.randn(w.size()) for w in weights]


def get_random_states(states):
    """Random Gaussian tensors matching ``state_dict()`` shapes (incl. BN stats)."""
    return [torch.randn(w.size()) for _k, w in states.items()]


def normalize_direction(direction, weights, norm="filter"):
    """Rescale ``direction`` relative to ``weights`` (one layer)."""
    if norm == "filter":
        for d, w in zip(direction, weights, strict=False):
            d.mul_(w.norm() / (d.norm() + 1e-10))
    elif norm == "layer":
        direction.mul_(weights.norm() / direction.norm())
    elif norm == "weight":
        direction.mul_(weights)
    elif norm == "dfilter":
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == "dlayer":
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction, weights, norm="filter", ignore="biasbn"):
    """Scale direction entries using ``weights``; zero 1-D tensors if ``ignore='biasbn'``."""
    assert len(direction) == len(weights)
    for d, w in zip(direction, weights, strict=False):
        if d.dim() <= 1:
            if ignore == "biasbn":
                d.fill_(0)
            else:
                d.copy_(w)
        else:
            normalize_direction(d, w, norm)


def normalize_directions_for_states(direction, states, norm="filter", ignore="ignore"):
    assert len(direction) == len(states)
    for d, (_k, w) in zip(direction, states.items(), strict=False):
        if d.dim() <= 1:
            if ignore == "biasbn":
                d.fill_(0)
            else:
                d.copy_(w)
        else:
            normalize_direction(d, w, norm)


def create_random_direction(net, dir_type="weights", ignore="biasbn", norm="filter"):
    """Random normalized direction in weight or state space (Li et al. loss landscapes)."""
    if dir_type == "weights":
        weights = get_weights(net)
        direction = get_random_weights(weights)
        normalize_directions_for_weights(direction, weights, norm, ignore)
    elif dir_type == "states":
        states = net.state_dict()
        direction = get_random_states(states)
        normalize_directions_for_states(direction, states, norm, ignore)
    else:
        raise ValueError(f"dir_type must be 'weights' or 'states', got {dir_type!r}")

    return direction
