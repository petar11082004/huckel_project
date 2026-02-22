"""
huckel.py

Utilities for building and analyzing Huckel (tight-binding) Hamiltonians for
conjugated pi systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _group_degeneracies(evals: np.ndarray, tol: float = 1e-6) -> dict[float, int]:
    """Group nearly equal eigenvalues and return multiplicities."""
    if evals.size == 0:
        return {}

    rounded = np.round(evals / tol) * tol
    unique_vals, counts = np.unique(rounded, return_counts=True)
    return {float(v): int(c) for v, c in zip(unique_vals, counts)}


def _build_matrix_from_edges(
    n: int,
    edges: Iterable[tuple[int, int]],
    alpha: float = 0.0,
    beta: float = -1.0,
) -> np.ndarray:
    """Construct an n x n Huckel matrix from an edge list."""
    if n <= 0:
        raise ValueError("n must be a positive integer")

    H = np.zeros((n, n), dtype=float)
    np.fill_diagonal(H, alpha)

    for i, j in edges:
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"edge ({i}, {j}) out of bounds for n={n}")
        if i == j:
            raise ValueError("self-loops are not allowed in Huckel adjacency")
        H[i, j] = beta
        H[j, i] = beta

    return H


def linear_polyene_matrix(n: int, alpha: float = 0.0, beta: float = -1.0) -> np.ndarray:
    """Huckel matrix for a linear polyene with n carbons."""
    edges = [(i, i + 1) for i in range(n - 1)]
    return _build_matrix_from_edges(n, edges, alpha=alpha, beta=beta)


def cyclic_polyene_matrix(n: int, alpha: float = 0.0, beta: float = -1.0) -> np.ndarray:
    """Huckel matrix for a cyclic polyene with n carbons."""
    if n < 3:
        raise ValueError("cyclic polyene requires n >= 3")
    edges = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
    return _build_matrix_from_edges(n, edges, alpha=alpha, beta=beta)


def naphthalene_matrix(alpha: float = 0.0, beta: float = -1.0) -> np.ndarray:
    """
    Huckel matrix for naphthalene (C10H8) using a fused-ring graph (10 carbons).

    Vertex labeling is an internal convention. The graph is two fused hexagons
    sharing one edge.
    """
    edges = [
        # First six-membered ring
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
        # Second ring fused along edge (4, 5)
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
    ]
    return _build_matrix_from_edges(10, edges, alpha=alpha, beta=beta)


def platonic_solid_edges(name: str) -> tuple[int, list[tuple[int, int]]]:
    """
    Return (n, edges) for 3-regular Platonic-solid graphs with carbon at vertices.

    Supported:
    - tetrahedron (4 vertices)
    - cube / hexahedron (8 vertices)
    - dodecahedron (20 vertices)

    Optional non-sp2 cases (still graph-theory interesting):
    - octahedron
    - icosahedron
    """
    key = name.strip().lower()

    if key in {"tetrahedron", "tetra"}:
        # K4
        n = 4
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return n, edges

    if key in {"cube", "hexahedron"}:
        # 3D hypercube graph: vertices 0..7 as 3-bit labels, edges differ by one bit.
        n = 8
        edges: list[tuple[int, int]] = []
        for v in range(n):
            for bit in (1, 2, 4):
                u = v ^ bit
                if v < u:
                    edges.append((v, u))
        return n, edges

    if key == "dodecahedron":
        # Standard 20-vertex dodecahedral graph represented as:
        # outer decagon + inner decagon + 10 matching spokes
        n = 20
        edges: list[tuple[int, int]] = []

        # Outer 10-cycle: 0..9
        edges += [(i, (i + 1) % 10) for i in range(10)]

        # Inner connections of generalized Petersen graph G(10,2), which is the
        # dodecahedral graph.
        inner = list(range(10, 20))
        for i in range(10):
            a = inner[i]
            b = inner[(i + 2) % 10]
            if a < b:
                edges.append((a, b))
            else:
                edges.append((b, a))

        # Spokes connect corresponding outer/inner vertices
        edges += [(i, 10 + i) for i in range(10)]

        # Remove duplicates introduced by canonical ordering
        edges = sorted(set(edges))
        return n, edges

    if key == "octahedron":
        # Octahedron graph = K_{2,2,2} (4-regular, not sp2)
        n = 6
        opposite = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if opposite[i] != j:
                    edges.append((i, j))
        return n, edges

    if key == "icosahedron":
        # One common indexing for icosahedral graph (5-regular, not sp2)
        n = 12
        edges = [
            (0, 1), (0, 4), (0, 5), (0, 7), (0, 10),
            (1, 2), (1, 5), (1, 6), (1, 8),
            (2, 3), (2, 6), (2, 7), (2, 9),
            (3, 4), (3, 7), (3, 8), (3, 10),
            (4, 5), (4, 8), (4, 9),
            (5, 9), (5, 11),
            (6, 8), (6, 10), (6, 11),
            (7, 10), (7, 11),
            (8, 9), (9, 11), (10, 11),
        ]
        return n, edges

    raise ValueError(f"Unknown Platonic solid '{name}'")


def platonic_solid_matrix(name: str, alpha: float = 0.0, beta: float = -1.0) -> np.ndarray:
    """Huckel matrix for a supported Platonic solid graph."""
    n, edges = platonic_solid_edges(name)
    return _build_matrix_from_edges(n, edges, alpha=alpha, beta=beta)


def analytic_linear_polyene_energies(
    n: int,
    alpha: float = 0.0,
    beta: float = -1.0,
) -> np.ndarray:
    """Analytic Huckel energies for a linear polyene."""
    k = np.arange(1, n + 1, dtype=float)
    energies = alpha + 2.0 * beta * np.cos(np.pi * k / (n + 1))
    return np.sort(energies)


def analytic_cyclic_polyene_energies(
    n: int,
    alpha: float = 0.0,
    beta: float = -1.0,
) -> np.ndarray:
    """Analytic Huckel energies for a cyclic polyene."""
    k = np.arange(n, dtype=float)
    energies = alpha + 2.0 * beta * np.cos(2.0 * np.pi * k / n)
    return np.sort(energies)


@dataclass
class HuckelResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    degeneracies: dict[float, int]


class HuckelSystem:
    """
    Represents a Huckel system and provides matrix construction and diagonalization.

    Backward-compatible constructor usage:
        HuckelSystem(n, "linear")
        HuckelSystem(n, "cyclic")

    Additional usage:
        HuckelSystem.from_matrix(H)
        HuckelSystem.from_edges(n, edges)
        HuckelSystem.naphthalene()
        HuckelSystem.platonic_solid("cube")
    """

    def __init__(
        self,
        n: int,
        system_type: str = "linear",
        alpha: float = 0.0,
        beta: float = -1.0,
    ):
        self.n = int(n)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.system_type = system_type.lower()
        self.matrix: np.ndarray | None = None
        self.eigenvalues: np.ndarray | None = None
        self.eigenvectors: np.ndarray | None = None
        self.degeneracies: dict[float, int] | None = None

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, label: str = "custom") -> "HuckelSystem":
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Huckel matrix must be symmetric")
        obj = cls(matrix.shape[0], system_type=label)
        obj.matrix = matrix.copy()
        return obj

    @classmethod
    def from_edges(
        cls,
        n: int,
        edges: Iterable[tuple[int, int]],
        alpha: float = 0.0,
        beta: float = -1.0,
        label: str = "custom",
    ) -> "HuckelSystem":
        obj = cls(n, system_type=label, alpha=alpha, beta=beta)
        obj.matrix = _build_matrix_from_edges(n, edges, alpha=alpha, beta=beta)
        return obj

    @classmethod
    def naphthalene(cls, alpha: float = 0.0, beta: float = -1.0) -> "HuckelSystem":
        obj = cls(10, system_type="naphthalene", alpha=alpha, beta=beta)
        obj.matrix = naphthalene_matrix(alpha=alpha, beta=beta)
        return obj

    @classmethod
    def platonic_solid(
        cls,
        name: str,
        alpha: float = 0.0,
        beta: float = -1.0,
    ) -> "HuckelSystem":
        n, _ = platonic_solid_edges(name)
        obj = cls(n, system_type=name, alpha=alpha, beta=beta)
        obj.matrix = platonic_solid_matrix(name, alpha=alpha, beta=beta)
        return obj

    def build_matrix(self) -> None:
        """Construct the Huckel Hamiltonian for linear/cyclic systems."""
        if self.system_type == "linear":
            self.matrix = linear_polyene_matrix(self.n, alpha=self.alpha, beta=self.beta)
        elif self.system_type == "cyclic":
            self.matrix = cyclic_polyene_matrix(self.n, alpha=self.alpha, beta=self.beta)
        else:
            if self.matrix is None:
                raise ValueError(
                    "Unsupported system_type for automatic construction. "
                    "Use from_edges/from_matrix/naphthalene/platonic_solid."
                )

    def calculate_mos(self, tol: float = 1e-6) -> HuckelResult:
        """Diagonalize the Huckel matrix and compute degeneracies."""
        if self.matrix is None:
            self.build_matrix()

        # Huckel Hamiltonians are symmetric; eigh is more stable than eig.
        vals, vecs = np.linalg.eigh(self.matrix)
        idx = np.argsort(vals)
        self.eigenvalues = vals[idx]
        self.eigenvectors = vecs[:, idx]
        self.degeneracies = _group_degeneracies(self.eigenvalues, tol=tol)
        return HuckelResult(self.eigenvalues, self.eigenvectors, self.degeneracies)

    def degree_sequence(self) -> list[int]:
        """Return graph degree sequence inferred from off-diagonal beta entries."""
        if self.matrix is None:
            self.build_matrix()
        assert self.matrix is not None
        off_diag = self.matrix.copy()
        np.fill_diagonal(off_diag, 0.0)
        # Count non-zero couplings. Works for default beta and general beta != 0.
        return sorted(np.count_nonzero(np.abs(off_diag) > 0.0, axis=1).tolist())

    def summary(self) -> None:
        """Print a compact summary of the system."""
        print("\nHuckel System Summary")
        print(f" Type: {self.system_type}")
        print(f" n = {self.n}, alpha = {self.alpha}, beta = {self.beta}")
        if self.matrix is None:
            print(" Matrix not yet built.")
            return
        print(" Hamiltonian:")
        print(self.matrix)
        if self.eigenvalues is None:
            print(" Eigenvalues not yet calculated.")
            return
        print(" Eigenvalues:", self.eigenvalues)
        print(" Degeneracies:", self.degeneracies)


def solve_huckel_matrix(matrix: np.ndarray, tol: float = 1e-6) -> HuckelResult:
    """Convenience function for diagonalizing an already-built Huckel matrix."""
    system = HuckelSystem.from_matrix(matrix)
    return system.calculate_mos(tol=tol)
