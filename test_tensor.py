"""Minimal sanity tests for the moment-of-inertia-tensor descriptor."""
import numpy as np

from tensor_prioritization import (
    eig_val,
    eig_val_weighted,
    distance_matrix,
    principal_moments,
    rotational_energy,
    angular_momentum,
    RESIDUE_COORDS,
    RESIDUE_WEIGHTS,
)


def test_eig_val_returns_three_eigenvalues():
    vals = eig_val("MALLIKARJUNA")
    assert len(vals) == 3


def test_eig_val_eigenvalues_nonnegative():
    # The inertia tensor is positive semi-definite, so eigenvalues are >= 0.
    vals = np.real(eig_val("ACDEFGHIKLMNPQRSTVWY"))
    assert np.all(vals >= -1e-9)


def test_residue_map_covers_standard_amino_acids():
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        assert aa in RESIDUE_COORDS


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    seqs = ["MKLV", "AACD", "PQRS"]
    desc = [eig_val(s) for s in seqs]
    dmat = distance_matrix(desc, desc, seqs, seqs).values
    assert np.allclose(dmat, dmat.T)
    assert np.allclose(np.diag(dmat), 0.0)


def test_weighted_descriptor_returns_three_eigenvalues():
    vals = eig_val_weighted("MALLIKARJUNA")
    assert len(vals) == 3


def test_residue_weights_cover_standard_amino_acids():
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        assert aa in RESIDUE_WEIGHTS


def test_principal_moments_sorted_and_nonnegative():
    moments = principal_moments("ACDEFGHIKLMNPQRSTVWY")
    assert len(moments) == 3
    assert np.all(np.diff(moments) >= -1e-9)     # ascending
    assert np.all(moments >= -1e-9)              # non-negative


def test_rotational_quantities_scale_as_expected():
    moments = principal_moments("MKLVACDE")
    # Doubling angular velocity: energy x4, angular momentum x2.
    e1 = rotational_energy(moments, omega=(1, 1, 1))
    e2 = rotational_energy(moments, omega=(2, 2, 2))
    l1 = angular_momentum(moments, omega=(1, 1, 1))
    l2 = angular_momentum(moments, omega=(2, 2, 2))
    assert np.isclose(e2, 4 * e1)
    assert np.isclose(l2, 2 * l1)
