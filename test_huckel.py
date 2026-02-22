import unittest

import numpy as np

from huckel import (
    HuckelSystem,
    analytic_cyclic_polyene_energies,
    analytic_linear_polyene_energies,
)


class TestHuckelSystem(unittest.TestCase):
    def test_linear_matrix_shape(self):
        h = HuckelSystem(5, "linear")
        h.build_matrix()
        self.assertEqual(h.matrix.shape, (5, 5))

    def test_cyclic_matrix_connection(self):
        h = HuckelSystem(4, "cyclic")
        h.build_matrix()
        self.assertEqual(h.matrix[0, 3], h.beta)
        self.assertEqual(h.matrix[3, 0], h.beta)

    def test_matrix_symmetry(self):
        h = HuckelSystem(6, "cyclic")
        h.build_matrix()
        self.assertTrue(np.allclose(h.matrix, h.matrix.T))

    def test_eigenvalues_count(self):
        h = HuckelSystem(6, "cyclic")
        h.calculate_mos()
        self.assertEqual(len(h.eigenvalues), h.n)

    def test_degeneracy_sum(self):
        h = HuckelSystem(6, "cyclic")
        h.calculate_mos()
        self.assertEqual(sum(h.degeneracies.values()), h.n)

    def test_known_butadiene_eigenvalues(self):
        h = HuckelSystem(4, "linear")
        h.calculate_mos()
        expected = analytic_linear_polyene_energies(4, alpha=h.alpha, beta=h.beta)
        self.assertTrue(np.allclose(expected, h.eigenvalues, atol=1e-6))

    def test_benzene_cyclic_analytic_energies(self):
        h = HuckelSystem(6, "cyclic")
        h.calculate_mos()
        expected = analytic_cyclic_polyene_energies(6, alpha=h.alpha, beta=h.beta)
        self.assertTrue(np.allclose(expected, h.eigenvalues, atol=1e-6))

    def test_benzene_degeneracies(self):
        h = HuckelSystem(6, "cyclic")
        h.calculate_mos()
        self.assertEqual(sorted(h.degeneracies.values()), [1, 1, 2, 2])

    def test_tetrahedron_spectrum_pattern(self):
        h = HuckelSystem.platonic_solid("tetrahedron")
        h.calculate_mos()
        self.assertEqual(h.n, 4)
        self.assertEqual(h.degree_sequence(), [3, 3, 3, 3])
        # For K4 adjacency spectrum is {3, -1, -1, -1}; beta = -1 flips sign.
        self.assertTrue(np.allclose(h.eigenvalues, np.array([-3.0, 1.0, 1.0, 1.0])))
        self.assertEqual(sorted(h.degeneracies.values()), [1, 3])

    def test_cube_degeneracies(self):
        h = HuckelSystem.platonic_solid("cube")
        h.calculate_mos()
        self.assertEqual(h.n, 8)
        self.assertEqual(h.degree_sequence(), [3] * 8)
        self.assertEqual(sorted(h.degeneracies.values()), [1, 1, 3, 3])

    def test_dodecahedron_graph_is_3_regular(self):
        h = HuckelSystem.platonic_solid("dodecahedron")
        h.build_matrix()
        self.assertEqual(h.matrix.shape, (20, 20))
        self.assertTrue(np.allclose(h.matrix, h.matrix.T))
        self.assertEqual(h.degree_sequence(), [3] * 20)

    def test_naphthalene_graph(self):
        h = HuckelSystem.naphthalene()
        h.build_matrix()  # no-op but harmless if already built
        self.assertEqual(h.matrix.shape, (10, 10))
        self.assertTrue(np.allclose(h.matrix, h.matrix.T))
        # Naphthalene has 11 C-C pi couplings -> 22 off-diagonal nonzero entries.
        off_diag_nz = np.count_nonzero(h.matrix - np.diag(np.diag(h.matrix)))
        self.assertEqual(off_diag_nz, 22)
        self.assertEqual(h.degree_sequence().count(3), 2)
        self.assertEqual(h.degree_sequence().count(2), 8)


if __name__ == "__main__":
    unittest.main()

