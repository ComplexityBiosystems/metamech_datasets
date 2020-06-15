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

from pathlib import Path
from metamech.lattice import Lattice
from metamech.actuator import Actuator
from metamech.io import read_lammps
from metamech.metropolis import Metropolis
from typing import Union
import numpy as np
import pandas as pd


def load_jammed_discs(
    method: str = "displacement",
    size: str = "small",
    configuration: int = 0,
    solver_tol: float = 1e-5,
) -> Actuator:
    if method not in ["displacement", "force"]:
        raise RuntimeError(f"Method not known")
    elif method == "displacement":
        return _load_jammed_discs_displacement(
            solver_tol=solver_tol,
            size=size,
            configuration=configuration
        )
    elif method == "force":
        return _load_jammed_discs_force(
            solver_tol=solver_tol,
            size=size,
            configuration=configuration
        )
    else:
        raise NotImplementedError(f"Method {method} known but not implemented")


def _load_jammed_discs_force(
    solver_tol: float = 1e-5,
    size: str = "small",
    configuration: int = 0,
) -> Actuator:
    lattice = _load_jammed_discs_lattice(
        size=size, configuration=configuration)

    # decide input and output
    input_nodes = []
    output_nodes = []
    frozen_nodes = []
    xs, ys = lattice._nodes_positions.T
    frozen_nodes = [
        i for i, y in enumerate(ys)
        if y < min(ys) + 0.10 * (max(ys) - min(ys))
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
    input_vectors = np.array([
        [0, -0.0015]
        for _ in input_nodes
    ])

    output_vectors = np.array([
        [2, 0]
        for _ in output_nodes
    ])

    actuator = Actuator(
        lattice=lattice,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        frozen_nodes=frozen_nodes,
        input_vectors=input_vectors,
        output_vectors=output_vectors,
        max_force=solver_tol,
        method="force",
        output_spring_stiffness=0.001,
        efficiency_agg_fun=np.sum
    )
    return actuator


def _load_jammed_discs_displacement(
    solver_tol: float = 1e-5,
    size: str = "small",
    configuration: int = 0,
    displacement_fraction: float = 0.01,
) -> Actuator:
    lattice = _load_jammed_discs_lattice(
        size=size, configuration=configuration)

    # decide input and output
    input_nodes = []
    output_nodes = []
    frozen_nodes = []
    xs, ys = lattice._nodes_positions.T
    frozen_nodes = [
        i for i, y in enumerate(ys)
        if y < min(ys) + 0.10 * (max(ys) - min(ys))
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


def _load_jammed_discs_lattice(size: str = "small", configuration: int = 0) -> Lattice:
    if size not in ["small", "large"]:
        raise RuntimeError("Size must be either 'small' or 'large'.")
    # only one large config available at the moment
    if size == "large":
        configuration = 3458

    input_lammps_file = Path(__file__).parent / \
        f"data/jammed_discs/{size}_size/conf_{configuration}.data"

    params = read_lammps(input_lammps_file)
    nodes_postions = params["nodes_positions"]
    edges_indices = params["edges_indices"]

    lattice = Lattice(
        nodes_positions=nodes_postions,
        edges_indices=edges_indices,
        linear_stiffness=10,
        angular_stiffness=0.2,
    )
    for edge in lattice._possible_edges:
        lattice.flip_edge(edge)

    # rescale edges to length 1
    average_edge_length = np.mean([
        edge.resting_length
        for edge in lattice._possible_edges
    ])
    nodes_postions /= average_edge_length
    lattice = Lattice(
        nodes_positions=nodes_postions,
        edges_indices=edges_indices,
        linear_stiffness=10,
        angular_stiffness=0.2,
    )
    for edge in lattice._possible_edges:
        lattice.flip_edge(edge)

    return lattice
