"""Threshold-aware, maximum-cardinality peak matching; no peak reuse."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_positions(left, right, max_decades=0.5, allowed_pairs=None):
    """Match positive times/frequencies, maximizing valid pairs before distance.

    Dummy columns allow unmatched peaks. A dummy costs more than the sum of
    all allowed distances, so an invalid match never displaces a valid one.
    Candidate degrees record resolution ambiguity, not a mechanism assignment.
    """
    a, b = np.asarray(left, float), np.asarray(right, float)
    if max_decades <= 0 or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Peak positions and matching tolerance must be positive")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Nonfinite peak position")
    distance = np.abs(np.log10(a[:, None] / b[None, :]))
    valid = distance <= max_decades
    if allowed_pairs is not None:
        allowed_pairs = np.asarray(allowed_pairs, bool)
        if allowed_pairs.shape != valid.shape:
            raise ValueError("Pair eligibility must match the position matrix")
        valid &= allowed_pairs
    matches = {}
    if len(a) and len(b):
        unmatched_cost = (len(a) + len(b) + 1) * (max_decades + 1)
        costs = np.full((len(a), len(b) + len(a)), unmatched_cost)
        costs[:, :len(b)] = np.where(valid, distance, 3 * unmatched_cost)
        rows, cols = linear_sum_assignment(costs)
        matches = {int(i): (int(j), float(distance[i, j]))
                   for i, j in zip(rows, cols) if j < len(b) and valid[i, j]}
    # Close but well-resolved peaks can have multiple geometric candidates.
    # Mark split/merge only where a connected neighborhood changes peak count.
    resolution_change = [False] * len(a)
    unseen = set(range(len(a)))
    while unseen:
        left_nodes, right_nodes = {min(unseen)}, set()
        while True:
            new_right = set(np.flatnonzero(valid[list(left_nodes)].any(axis=0))) if len(b) else set()
            new_left = set(np.flatnonzero(valid[:, list(new_right)].any(axis=1))) if new_right else left_nodes
            if new_left == left_nodes and new_right == right_nodes:
                break
            left_nodes, right_nodes = new_left, new_right
        for i in left_nodes:
            resolution_change[i] = bool(right_nodes and len(left_nodes) != len(right_nodes))
        unseen -= left_nodes
    return {
        "matches": matches,
        "left_candidate_counts": valid.sum(axis=1).astype(int).tolist(),
        "right_candidate_counts": valid.sum(axis=0).astype(int).tolist(),
        "ambiguous_left": [bool(valid[i].sum() > 1 or np.any(valid.sum(axis=0)[valid[i]] > 1))
                           for i in range(len(a))],
        "resolution_change_left": resolution_change,
        "matching_rule": "maximum-cardinality then minimum log-distance; one-to-one; no extrapolation",
    }
