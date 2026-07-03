"""Moment-of-inertia-tensor descriptors for protein sequences.

Reference implementation of:
    Thummadi NB, Mallikarjuna T, Vindal V, Manimaran P.
    "Prioritizing the candidate genes related to cervical cancer using the
    moment of inertia tensor." Proteins. 2022;90(2):363-371.
    doi:10.1002/prot.26226

The functions here mirror the notebook so the method can be imported and reused
as a library.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import eig

# Fixed 3D coordinates for each residue: a point on the unit circle in the x-y
# plane, with z in {-1, +1} encoding a physicochemical property.
RESIDUE_COORDS = {
    "A": (0.9511, 0.3090, -1), "B": (0.9511, 0.3090, -1),
    "C": (0.8090, 0.5878, -1), "D": (-0.9511, -0.3090, 1),
    "E": (0.3090, -0.9511, 1), "F": (-0.3090, 0.9511, 1),
    "G": (0.5878, -0.8090, -1), "H": (0.8090, -0.5878, 1),
    "I": (-0.5878, 0.8090, -1), "J": (-0.5878, 0.8090, -1),
    "K": (-0.8090, -0.5878, 1), "L": (-0.8090, 0.5878, -1),
    "M": (0.5878, 0.8090, 1), "N": (-0.5878, -0.8090, -1),
    "O": (-0.5878, -0.8090, -1), "P": (0.3090, 0.9511, -1),
    "Q": (0.9511, -0.3090, 1), "R": (-0.3090, -0.9511, 1),
    "S": (0.0, -1.0, -1), "T": (1.0, 0.0, -1),
    "U": (1.0, 0.0, -1), "V": (0.0, 1.0, -1),
    "W": (-0.9511, 0.3090, 1), "X": (-0.9511, 0.3090, 1),
    "Y": (-1.0, 0.0, 1), "Z": (-1.0, 0.0, 1),
}


def eig_val(sequence) -> np.ndarray:
    """Return the three eigenvalues of the moment-of-inertia tensor of a sequence."""
    xs, ys, zs = [], [], []
    for residue in sequence:
        x, y, z = RESIDUE_COORDS[residue]
        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.asarray(xs) - np.mean(xs)
    ys = np.asarray(ys) - np.mean(ys)
    zs = np.asarray(zs) - np.mean(zs)

    ixx = np.sum(ys ** 2 + zs ** 2)
    iyy = np.sum(xs ** 2 + zs ** 2)
    izz = np.sum(xs ** 2 + ys ** 2)
    ixy = np.sum(xs * ys)
    iyz = np.sum(ys * zs)
    ixz = np.sum(xs * zs)

    inertia = np.array([
        [ixx, -ixy, -ixz],
        [-ixy, iyy, -iyz],
        [-ixz, -iyz, izz],
    ])
    eigenvalues, _ = eig(inertia)
    return eigenvalues


def retrieve(accession_ids, email: str):
    """Fetch FASTA records from NCBI and return their eigenvalue descriptors.

    Requires Biopython and a valid contact email for NCBI Entrez.
    """
    from Bio import Entrez, SeqIO

    Entrez.email = email
    handle = Entrez.efetch(db="protein", id=accession_ids,
                           rettype="fasta", retmode="text")
    records = list(SeqIO.parse(handle, "fasta"))
    handle.close()
    return [eig_val(rec.seq) for rec in records]


def distance_matrix(row_descriptors, col_descriptors,
                    row_labels=None, col_labels=None) -> pd.DataFrame:
    """Euclidean distance matrix between two sets of eigenvalue descriptors."""
    dist = np.array([[np.linalg.norm(r - c) for c in col_descriptors]
                     for r in row_descriptors])
    return pd.DataFrame(dist, index=row_labels, columns=col_labels)


def prioritize(dmat: pd.DataFrame, threshold_frac: float = 0.01,
               min_frequency: int = 6) -> pd.DataFrame:
    """Threshold the distance matrix and rank candidates by number of hits."""
    threshold = threshold_frac * (dmat.to_numpy().max() - dmat.to_numpy().min())
    binary = (dmat > threshold).astype(int)      # 0 = similar (hit), 1 = far
    frequency = (1 - binary).sum(axis=0)         # hits per candidate (column)
    ranked = frequency[frequency >= min_frequency].sort_values(ascending=False)
    return ranked.rename("hits").to_frame()


# ---------------------------------------------------------------------------
# Mass-weighted variant and rotational quantities
#
# The lung-cancer study (Dwivedi, Thippana, Manimaran, Vindal, bioRxiv 2024,
# doi:10.1101/2024.10.18.619023) extends the method by weighting the moment of
# inertia tensor with each residue's molecular mass, and by discussing the
# rotational energy / angular momentum of the resulting rigid body. The
# functions below provide the mass-weighted descriptor and the standard
# rigid-body rotational quantities built on the principal moments of inertia.
# ---------------------------------------------------------------------------

# Average residue (monomer) molecular weights in daltons.
RESIDUE_WEIGHTS = {
    "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
    "E": 129.12, "Q": 128.13, "G": 57.05, "H": 137.14, "I": 113.16,
    "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18, "P": 97.12,
    "S": 87.08, "T": 101.10, "W": 186.21, "Y": 163.18, "V": 99.13,
    # Ambiguity codes reuse the coordinate map; give them a neutral mass.
    "B": 114.60, "Z": 128.62, "X": 128.16, "U": 150.04, "J": 113.16,
    "O": 128.17,
}


def eig_val_weighted(sequence, weights=None):
    """Mass-weighted moment-of-inertia eigenvalues of a sequence.

    Like :func:`eig_val`, but the centre of mass and every tensor component are
    weighted by the residue molecular mass (``RESIDUE_WEIGHTS`` by default),
    following the lung-cancer variant of the method. Coordinates use the same
    ``RESIDUE_COORDS`` scheme; confirm the coordinate/weight table matches your
    manuscript before reporting results.
    """
    if weights is None:
        weights = RESIDUE_WEIGHTS

    coords = np.array([RESIDUE_COORDS[r] for r in sequence], dtype=float)
    m = np.array([weights[r] for r in sequence], dtype=float)

    # Mass-weighted centre of mass, then shift the cloud onto it.
    com = (coords * m[:, None]).sum(axis=0) / m.sum()
    x, y, z = (coords - com).T

    ixx = np.sum(m * (y ** 2 + z ** 2))
    iyy = np.sum(m * (x ** 2 + z ** 2))
    izz = np.sum(m * (x ** 2 + y ** 2))
    ixy = np.sum(m * x * y)
    iyz = np.sum(m * y * z)
    ixz = np.sum(m * x * z)

    inertia = np.array([
        [ixx, -ixy, -ixz],
        [-ixy, iyy, -iyz],
        [-ixz, -iyz, izz],
    ])
    eigenvalues, _ = eig(inertia)
    return eigenvalues


def principal_moments(sequence, weighted=True):
    """Return the three principal moments of inertia, sorted ascending."""
    vals = eig_val_weighted(sequence) if weighted else eig_val(sequence)
    return np.sort(np.real(vals))


def rotational_energy(moments, omega=(1.0, 1.0, 1.0)):
    """Rigid-body rotational kinetic energy E = 1/2 * sum(I_i * omega_i^2).

    ``moments`` are the principal moments of inertia (e.g. from
    :func:`principal_moments`) and ``omega`` is the angular velocity about each
    principal axis. ``omega`` defaults to unit rotation about all axes; set it
    to match the convention used in your manuscript.
    """
    I = np.asarray(moments, dtype=float)
    w = np.asarray(omega, dtype=float)
    return 0.5 * np.sum(I * w ** 2)


def angular_momentum(moments, omega=(1.0, 1.0, 1.0)):
    """Magnitude of the rigid-body angular momentum |L|, with L_i = I_i * omega_i.

    See :func:`rotational_energy` for the meaning of ``moments`` and ``omega``.
    """
    I = np.asarray(moments, dtype=float)
    w = np.asarray(omega, dtype=float)
    return float(np.linalg.norm(I * w))
