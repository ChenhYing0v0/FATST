"""Projective Local-Global Operator readouts for StageC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn


PLGO_SERIES_LENGTH = 720
PLGO_GLOBAL_RANK = 16
PLGO_LATENT_WIDTH = 256
PLGO_DESCRIPTOR_DIM = 8
PLGO_PERMUTATION_SEED = 7101
PLGO_RANDOM_DESCRIPTOR_SEED = 7102


@dataclass(frozen=True)
class Atom:
    """Metadata for one RGNB synthesis atom."""

    kind: str
    depth: int
    start: int
    end: int


@dataclass
class _Node:
    start: int
    end: int
    depth: int
    scaling: torch.Tensor
    detail: torch.Tensor
    left: _Node | None = None
    right: _Node | None = None


def _dct_prototypes(length: int, rank: int) -> torch.Tensor:
    time = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    frequency = torch.arange(rank, dtype=torch.float64).unsqueeze(0)
    basis = torch.cos(math.pi * (time + 0.5) * frequency / length)
    basis[:, 0] *= math.sqrt(1.0 / length)
    if rank > 1:
        basis[:, 1:] *= math.sqrt(2.0 / length)
    return basis


def _restricted_coordinates(
    total_length: int,
    start: int,
    end: int,
    global_rank: int,
) -> torch.Tensor:
    size = end - start
    dimension = min(size, global_rank)
    if size <= global_rank:
        return torch.eye(size, dtype=torch.float64)
    time = torch.arange(start, end, dtype=torch.float64)
    coordinate = torch.cos(math.pi * (time + 0.5) / total_length)
    lower = coordinate.min()
    upper = coordinate.max()
    normalized = 2.0 * (coordinate - lower) / (upper - lower) - 1.0
    columns = [torch.ones_like(normalized)]
    if dimension > 1:
        columns.append(normalized)
    for _degree in range(2, dimension):
        columns.append(2.0 * normalized * columns[-1] - columns[-2])
    return torch.stack(columns, dim=1)


def _restricted_span(
    total_length: int,
    start: int,
    end: int,
    global_rank: int,
) -> torch.Tensor:
    coordinates = _restricted_coordinates(
        total_length,
        start,
        end,
        global_rank,
    )
    basis, _upper = torch.linalg.qr(coordinates, mode="reduced")
    return basis


def _block_diagonal(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(
        left.shape[0] + right.shape[0],
        left.shape[1] + right.shape[1],
        dtype=left.dtype,
    )
    result[: left.shape[0], : left.shape[1]] = left
    result[left.shape[0] :, left.shape[1] :] = right
    return result


def _build_node(
    total_length: int,
    global_rank: int,
    start: int,
    end: int,
    depth: int,
) -> _Node:
    scaling = _restricted_span(total_length, start, end, global_rank)
    if end - start == 1:
        return _Node(
            start=start,
            end=end,
            depth=depth,
            scaling=scaling,
            detail=torch.empty((1, 0), dtype=torch.float64),
        )
    middle = start + (end - start) // 2
    left = _build_node(total_length, global_rank, start, middle, depth + 1)
    right = _build_node(total_length, global_rank, middle, end, depth + 1)
    child_scaling = _block_diagonal(left.scaling, right.scaling)
    coordinates = child_scaling.T @ scaling
    _u, _singular_values, vh = torch.linalg.svd(
        coordinates.T,
        full_matrices=True,
    )
    complement = vh[scaling.shape[1] :].T
    return _Node(
        start=start,
        end=end,
        depth=depth,
        scaling=scaling,
        detail=child_scaling @ complement,
        left=left,
        right=right,
    )


def _collect_details(
    node: _Node,
    total_length: int,
    columns: list[torch.Tensor],
    atoms: list[Atom],
) -> None:
    for index in range(node.detail.shape[1]):
        column = torch.zeros(total_length, dtype=node.detail.dtype)
        column[node.start : node.end] = node.detail[:, index]
        columns.append(column)
        atoms.append(
            Atom(
                kind="detail",
                depth=node.depth,
                start=node.start,
                end=node.end,
            )
        )
    if node.left is not None and node.right is not None:
        _collect_details(node.left, total_length, columns, atoms)
        _collect_details(node.right, total_length, columns, atoms)


@lru_cache(maxsize=8)
def _cached_restricted_global_nested_basis(
    length: int,
    global_rank: int,
) -> tuple[torch.Tensor, tuple[Atom, ...]]:
    """Construct the square Restricted-Global Nested Basis."""
    prototypes = _dct_prototypes(length, global_rank)
    root = _build_node(length, global_rank, 0, length, 0)
    columns = [root.scaling[:, index] for index in range(global_rank)]
    atoms = [
        Atom(kind="global", depth=-1, start=0, end=length)
        for _index in range(global_rank)
    ]
    _collect_details(root, length, columns, atoms)
    synthesis = torch.stack(columns, dim=1)
    if synthesis.shape != (length, length) or len(atoms) != length:
        raise RuntimeError("RGNB construction did not produce a square basis")
    return synthesis, tuple(atoms)


def restricted_global_nested_basis(
    length: int,
    global_rank: int,
) -> tuple[torch.Tensor, list[Atom]]:
    """Return an isolated copy of the cached RGNB construction."""
    synthesis, atoms = _cached_restricted_global_nested_basis(length, global_rank)
    return synthesis.clone(), list(atoms)


def canonical_atom_descriptors(atoms: list[Atom]) -> torch.Tensor:
    """Encode RGNB geometry without a requested-horizon feature."""
    length = atoms[0].end
    max_depth = max((atom.depth for atom in atoms), default=0)
    groups: dict[tuple[str, int, int, int], list[int]] = {}
    for index, atom in enumerate(atoms):
        key = (atom.kind, atom.depth, atom.start, atom.end)
        groups.setdefault(key, []).append(index)
    rows = []
    for index, atom in enumerate(atoms):
        key = (atom.kind, atom.depth, atom.start, atom.end)
        group = groups[key]
        order = group.index(index)
        order_scale = max(len(group) - 1, 1)
        interval_length = atom.end - atom.start
        rows.append(
            [
                float(atom.kind == "global"),
                float(atom.kind == "detail"),
                atom.start / length,
                atom.end / length,
                interval_length / length,
                0.0 if atom.depth < 0 else atom.depth / max(max_depth, 1),
                order / order_scale,
                len(group) / length,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def descriptor_family(
    canonical: torch.Tensor,
    family: str,
    permutation_seed: int = PLGO_PERMUTATION_SEED,
    random_seed: int = PLGO_RANDOM_DESCRIPTOR_SEED,
) -> torch.Tensor:
    """Return canonical, permuted, or moment-matched random descriptors."""
    if family == "geo":
        return canonical.clone()
    if family == "perm":
        generator = torch.Generator(device="cpu").manual_seed(permutation_seed)
        return canonical[torch.randperm(canonical.shape[0], generator=generator)]
    if family != "random":
        raise ValueError(f"unsupported descriptor family: {family}")
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    random = torch.randn(canonical.shape, generator=generator)
    canonical_mean = canonical.mean(dim=0, keepdim=True)
    canonical_std = canonical.std(dim=0, unbiased=False, keepdim=True)
    random = (random - random.mean(dim=0, keepdim=True)) / random.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    ).clamp_min(1e-8)
    return random * canonical_std + canonical_mean


class PLGOPAFReadout(nn.Module):
    """Generate projective RGNB coefficients from a shared history latent."""

    def __init__(
        self,
        readout_dim: int,
        descriptor_name: str,
        trunk_width: int,
        series_length: int = PLGO_SERIES_LENGTH,
        global_rank: int = PLGO_GLOBAL_RANK,
        latent_width: int = PLGO_LATENT_WIDTH,
        permutation_seed: int = PLGO_PERMUTATION_SEED,
        random_seed: int = PLGO_RANDOM_DESCRIPTOR_SEED,
    ) -> None:
        super().__init__()
        synthesis, atoms = restricted_global_nested_basis(
            series_length,
            global_rank,
        )
        canonical = canonical_atom_descriptors(atoms)
        descriptors = descriptor_family(
            canonical,
            descriptor_name,
            permutation_seed,
            random_seed,
        )
        self.series_length = series_length
        self.global_rank = global_rank
        self.latent_width = latent_width
        self.trunk_width = trunk_width
        self.descriptor_name = descriptor_name
        self.branch = nn.Linear(readout_dim, latent_width)
        self.trunk = nn.Sequential(
            nn.Linear(PLGO_DESCRIPTOR_DIM, trunk_width),
            nn.Tanh(),
            nn.Linear(trunk_width, latent_width),
        )
        self.coefficient_bias = nn.Parameter(torch.zeros(series_length))
        self.register_buffer("basis_rows", synthesis.T.float())
        self.register_buffer("descriptors", descriptors)
        self.register_buffer(
            "atom_starts",
            torch.tensor([atom.start for atom in atoms], dtype=torch.long),
        )
        self.register_buffer(
            "atom_group_ids",
            torch.tensor(
                [0 if atom.kind == "global" else atom.depth + 1 for atom in atoms],
                dtype=torch.long,
            ),
        )

    def atom_features(self, indices: torch.Tensor | None = None) -> torch.Tensor:
        descriptors = self.descriptors if indices is None else self.descriptors[indices]
        return self.trunk(descriptors)

    def coefficients(
        self,
        hidden: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.branch(hidden)
        atom_features = self.atom_features(indices)
        bias = self.coefficient_bias if indices is None else self.coefficient_bias[indices]
        return torch.einsum("bck,nk->bcn", latent, atom_features) + bias

    def active_indices(self, horizon: int) -> torch.Tensor:
        if horizon <= 0 or horizon > self.series_length:
            raise ValueError("horizon must lie in [1, series_length]")
        return torch.nonzero(self.atom_starts < horizon, as_tuple=False).flatten()

    def latent_from_patch_blocks(
        self,
        hidden: torch.Tensor,
        patch_num: int,
        d_model: int,
    ) -> torch.Tensor:
        """Exact patch-explicit rewrite of the flattened branch projection."""
        if patch_num * d_model != hidden.shape[-1]:
            raise ValueError("patch shape does not match flattened history width")
        patches = hidden.reshape(*hidden.shape[:-1], patch_num, d_model)
        blocks = self.branch.weight.reshape(self.latent_width, patch_num, d_model)
        return torch.einsum("bcpd,kpd->bck", patches, blocks) + self.branch.bias

    def forward(
        self,
        hidden: torch.Tensor,
        target_prefix: int | None = None,
    ) -> torch.Tensor:
        horizon = self.series_length if target_prefix is None else int(target_prefix)
        active = self.active_indices(horizon)
        coefficients = self.coefficients(hidden, active)
        basis = self.basis_rows[active, :horizon].to(dtype=hidden.dtype)
        output = torch.einsum("bcn,nh->bch", coefficients, basis)
        return output.permute(0, 2, 1)
