"""Self-contained, offline example of the moment-of-inertia-tensor method.

Runs without any network access: it encodes a small set of protein sequences into
their eigenvalue descriptors, builds a Euclidean distance matrix, and saves a
heatmap + dendrogram to docs/example_output.png.

Run from the repository root:
    python examples/run_example.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend so it works without a display
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Make the src/ package importable when run from the repo root.
sys.path.insert(0, os.path.dirname(__file__))
from tensor_prioritization import eig_val, distance_matrix  # noqa: E402

# A few short protein fragments in two loose families (globin-like vs. serine
# protease-like). Real accession IDs would be fetched with retrieve(); these are
# inlined so the example needs no NCBI access.
SEQUENCES = {
    "GLB-1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF",
    "GLB-2": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFG",
    "GLB-3": "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRF",
    "GLB-4": "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKF",
    "PRT-1": "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQ",
    "PRT-2": "IVGGYTCAENSVPYQVSLNSGSHFCGGSLISEQWVVSAAHCYKTRIQ",
    "PRT-3": "IVNGEEAVPGSWPWQVSLQDKTGFHFCGGSLINENWVVTAAHCGVTT",
    "PRT-4": "IVGGRPCEKNSHPWQVALYHFSTFQCGGVLVNPKWVLTAAHCKNDNY",
}


def main():
    labels = list(SEQUENCES)
    descriptors = [eig_val(seq) for seq in SEQUENCES.values()]

    dmat = distance_matrix(descriptors, descriptors, labels, labels)
    print("Distance matrix:\n", dmat.round(2), "\n")

    # Hierarchical clustering from the (symmetric) distance matrix.
    linkage_matrix = linkage(squareform(dmat.values, checks=False), method="average")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    im = ax1.imshow(dmat.values, cmap="viridis")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=90)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_title("Euclidean distance matrix")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    dendrogram(linkage_matrix, labels=labels, ax=ax2)
    ax2.set_title("Dendrogram (average linkage)")
    ax2.set_ylabel("Distance")

    fig.tight_layout()

    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "example_output.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Figure saved to {os.path.normpath(out_path)}")


if __name__ == "__main__":
    main()
