# Codes used for the paper:
# "Automatic Design of Mechanical Metamaterial Actuators"
# by S. Bonfanti, R. Guerra, F. Font-Clos, R. Rayneau-Kirkhope, S. Zapperi
# Center for Complexity and Biosystems, University of Milan
# (c) University of Milan
#
#
######################################################################
#
# End User License Agreement (EULA)
# Your access to and use of the downloadable code (the "Code") is subject
# to a non-exclusive,  revocable, non-transferable,  and limited right to
# use the Code for the exclusive purpose of undertaking academic,
# governmental, or not-for-profit research. Use of the Code or any part
# thereof for commercial purposes is strictly prohibited in the absence
# of a Commercial License Agreement from the University of Milan. For
# information contact the Technology Transfer Office of the university
# of Milan (email: tto@unimi.it)
#
#######################################################################

from metamech.actuator import Actuator
from metamech.io import read_lammps
from metamech.lattice import Lattice
from pathlib import Path

import numpy as np

AVAILABLE_CRANE_SIZES = [8, 10, 12, 14, 16, 18, 20]


def load_crane_scaling(
    size: int = 8,
    displacement_fraction: float = 0.01,
    solver_tol: float = 1e-5,
) -> Actuator:
    """
    Load crane actuator for scaling analysis.

    Offers a set of initial configs on a regular
    lattice of increasing size and preconfigured
    input/output vectors, suitable for scaling
    analysis.

    Parameters
    ----------
    size : int, optional
        Number of nodes of side of lattice, by default 8
    solver_tol : float, optional
        Tolerance (max force) in FIRE algorithm, by default 1e-5
    displacement_fraction : float, optional
        Set input displacement to a fixed fraction of the length
        of the lattice in the y direction, by default 0.01.

    Returns
    -------
    : Actuator
        Preconfigured crane actuator.
    """
    # make sure size is available
    if size not in AVAILABLE_CRANE_SIZES:
        raise RuntimeError(
            f"Size {size} is not available. Available sizes are", AVAILABLE_CRANE_SIZES)
    path_to_initial_config = Path(__file__).parent / \
        f"data/crane_scaling/conf{size}.data"
    params = read_lammps(path_to_initial_config)
    nodes_postions = params["nodes_positions"]
    edges_indices = params["edges_indices"]

    # define input, output and frozen nodes
    xs, ys, _ = nodes_postions.T
    frozen_nodes = [
        i for i, y in enumerate(ys)
        if y == min(ys)
    ]
    input_nodes = [
        i for i, (x, y) in enumerate(zip(xs, ys))
        if
        (x > min(xs) + 0.8 * (max(xs) - min(xs))) and
        (y > min(ys) + 0.8 * (max(ys) - min(ys)))
    ]
    output_nodes = [
        i for i, (x, y) in enumerate(zip(xs, ys))
        if
        (x < min(xs) + 0.2 * (max(xs) - min(xs))) and
        (y > min(ys) + 0.8 * (max(ys) - min(ys)))
    ]
    # input and output vectors
    # displacement is 1% of y range
    input_vectors = np.array([
        [0, -displacement_fraction * (max(ys) - min(ys))]
        for _ in input_nodes
    ])

    output_vectors = np.array([
        [-1, 0]
        for _ in output_nodes
    ])

    lattice = Lattice(
        nodes_positions=nodes_postions,
        edges_indices=edges_indices,
        linear_stiffness=10,
        angular_stiffness=0.2,
    )
    for edge in lattice._possible_edges:
        lattice.flip_edge(edge)

    actuator = Actuator(
        lattice=lattice,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        frozen_nodes=frozen_nodes,
        input_vectors=input_vectors,
        output_vectors=output_vectors,
        max_force=solver_tol,
        method="displacement",
    )
    return actuator
