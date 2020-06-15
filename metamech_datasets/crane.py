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


def load_crane(
        method: str = "displacement",
        solver_tol: float = 1e-5
) -> Actuator:
    if method not in ["displacement", "force"]:
        raise RuntimeError(f"Method not known")
    elif method == "displacement":
        return _load_crane_displacement(
            solver_tol=solver_tol
        )
    elif method == "force":
        return _load_crane_force(
            solver_tol=solver_tol
        )
    else:
        raise NotImplementedError(f"Method {method} known but not implemented")


def _load_crane_force(solver_tol=1e-5) -> Actuator:
    path_to_initial_config = Path(__file__).parent / \
        "data/crane/crane_initial_config.lammps"

    params = read_lammps(path_to_initial_config)
    nodes_postions = params["nodes_positions"]
    edges_indices = params["edges_indices"]
    input_nodes = params["input_nodes"]
    output_nodes = params["output_nodes"]
    frozen_nodes = params["frozen_nodes"]

    lattice = Lattice(
        nodes_positions=nodes_postions,
        edges_indices=edges_indices,
        linear_stiffness=10,
        angular_stiffness=0.2,
    )
    for edge in lattice._possible_edges:
        lattice.flip_edge(edge)

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


def _load_crane_displacement(solver_tol=1e-5) -> Actuator:
    input_lammps_file = Path(__file__).parent / \
        "data/crane/crane_initial_config.lammps"

    params = read_lammps(input_lammps_file)
    nodes_postions = params["nodes_positions"]
    edges_indices = params["edges_indices"]
    input_nodes = params["input_nodes"]
    output_nodes = params["output_nodes"]
    frozen_nodes = params["frozen_nodes"]

    lattice = Lattice(
        nodes_positions=nodes_postions,
        edges_indices=edges_indices,
        linear_stiffness=10,
        angular_stiffness=0.2,
    )
    for edge in lattice._possible_edges:
        lattice.flip_edge(edge)

    # input and output vectors
    input_vectors = np.array([
        [0, -np.sqrt(3) / 4]
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
        method="displacement"
    )
    return actuator
