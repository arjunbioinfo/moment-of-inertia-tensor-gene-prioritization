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
