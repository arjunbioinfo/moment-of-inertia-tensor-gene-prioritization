"""Minimal sanity tests for the moment-of-inertia-tensor descriptor."""
import numpy as np

from tensor_prioritization import eig_val, distance_matrix, RESIDUE_COORDS


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
