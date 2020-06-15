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

from metamech.metropolis import Metropolis
from metamech.actuator import Actuator
import numpy as np
from metamech.io import read_lammps
from metamech.lattice import Lattice
import pandas as pd

from typing import Callable

from pathlib import Path

path_to_initial_config = Path(__file__).parent / \
    "data/plier/plier_initial_config.data"
path_to_human_solution = Path(__file__).parent / \
    "data/plier/plier_human_solution.data"
path_to_half_initial_config = Path(__file__).parent / \
    "data/plier/half_plier_initial_config.data"


def load_plier(
    method: str = "force",
    solver_tol: float = 1e-5,
    configuration: str = "full"
) -> Actuator:
    if method not in ["displacement", "force"]:
        raise RuntimeError(f"Method not known")
    elif method == "displacement":
        return _load_plier_displacement(
            solver_tol=solver_tol,
            configuration=configuration
        )
    elif method == "force":
        return _load_plier_force(
            solver_tol=solver_tol,
            configuration=configuration
        )
    else:
        raise NotImplementedError(f"Method {method} known but not implemented")


def _load_plier_displacement(configuration: str, solver_tol: float = 1e-5) -> Actuator:
    raise NotImplementedError


def _load_plier_force(
    configuration: str,
    solver_tol: float = 1e-5,
    force_norm: float = 0.01,
    out_spring_stiffness: float = 10,
    efficiency_agg_fun: Callable = np.sum,
) -> Actuator:
    assert configuration in ["human", "half_full"]

    # load the lattice
    lattice = _load_plier_lattice(configuration=configuration)

    if configuration == "half_full":
        # find input and output nodes
        input_coords = np.array([
            [20.5, 19.9186],
            [21.5, 19.9186],
            [22.5, 19.9186],
            [23.5, 19.9186],
            [24.5, 19.9186],
            [25.5, 19.9186]
        ])

        output_coords = np.array([
            [1, 12.1244],
            [2, 12.1244],
            [3, 12.1244]
        ])

        frozen_coords = np.array([
            [17, 10.3923],
        ])

        frozen_y_coords = np.array([
            [12, 10.3923],
            [13, 10.3923],
            [14, 10.3923],
            [15, 10.3923],
            [16, 10.3923],
            [18, 10.3923],
            [19, 10.3923],
            [20, 10.3923],
            [21, 10.3923],
            [22, 10.3923],
            [23, 10.3923],
            [24, 10.3923],
            [25, 10.3923],
            [26, 10.3923],
        ])
        frozen_y_nodes = []
        for (x, y) in frozen_y_coords:
            for node in lattice.nodes:
                if np.abs(x - node.x) < 0.01 and np.abs(y - node.y) < 0.01:
                    frozen_y_nodes.append(node.label)

        input_vectors = force_norm * np.array([
            [0, -1],
            [0, -1],
            [0, -1],
            [0, -1],
            [0, -1],
            [0, -1],
        ])

        output_vectors = np.array([
            [0.85, -2],
            [0.85, -2],
            [0.85, -2],
        ])

    elif configuration == "human":
        input_coords = np.array([
            [20.5, 0.86603],
            [21.5, 0.86603],
            [22.5, 0.86603],
            [23.5, 0.86603],
            [24.5, 0.86603],
            [25.5, 0.86603],
            [20.5, 19.9186],
            [21.5, 19.9186],
            [22.5, 19.9186],
            [23.5, 19.9186],
            [24.5, 19.9186],
            [25.5, 19.9186]
        ])

        output_coords = np.array([
            [1, 8.66025],
            [2, 8.66025],
            [3, 8.66025],
            [1, 12.1244],
            [2, 12.1244],
            [3, 12.1244]
        ])

        frozen_coords = np.array([
            [17, 10.3923],
        ])

        frozen_y_nodes = []
        input_vectors = force_norm * np.array([
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, -1],
            [0, -1],
            [0, -1],
            [0, -1],
            [0, -1],
            [0, -1],
        ])

        output_vectors = np.array([
            [0.85, 2],
            [0.85, 2],
            [0.85, 2],
            [0.85, -2],
            [0.85, -2],
            [0.85, -2],
        ])

    # common stuff
    input_nodes = []
    for (x, y) in input_coords:
        for node in lattice.nodes:
            if np.abs(x - node.x) < 0.01 and np.abs(y - node.y) < 0.01:
                input_nodes.append(node.label)

    output_nodes = []
    for (x, y) in output_coords:
        for node in lattice.nodes:
            if np.abs(x - node.x) < 0.01 and np.abs(y - node.y) < 0.01:
                output_nodes.append(node.label)

    frozen_nodes = []
    for (x, y) in frozen_coords:
        for node in lattice.nodes:
            if np.abs(x - node.x) < 0.01 and np.abs(y - node.y) < 0.01:
                frozen_nodes.append(node.label)
            if configuration == "half_full":
                if node.y < 10.39:
                    frozen_nodes.append(node.label)

    output_vectors = (output_vectors.T /
                      np.sqrt(np.sum(output_vectors ** 2, axis=1))).T

    actuator = Actuator(
        lattice=lattice,
        frozen_nodes=frozen_nodes,
        frozen_y_nodes=frozen_y_nodes,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        input_vectors=input_vectors,
        output_vectors=output_vectors,
        output_spring_stiffness=out_spring_stiffness,
        method="force",
        max_force=solver_tol,
        efficiency_agg_fun=efficiency_agg_fun,
    )
    return actuator


def _load_plier_lattice(configuration: str = "full") -> Lattice:
    assert configuration in ["full", "human", "half_full"]

    full = read_lammps(path_to_initial_config)["nodes_positions"].T[:2].T
    _full_edges = read_lammps(path_to_initial_config)["edges_indices"]

    full_to_standard = {
        v: k
        for k, v
        in dict(enumerate(np.unique(_full_edges.reshape(-1)))).items()
    }
    full_edges = np.vectorize(full_to_standard.get)(_full_edges)

    human = read_lammps(path_to_human_solution)["nodes_positions"].T[:2].T
    human_edges = read_lammps(path_to_human_solution)["edges_indices"]

    half_full = read_lammps(path_to_half_initial_config)[
        "nodes_positions"].T[:2].T
    half_full_edges = read_lammps(path_to_half_initial_config)["edges_indices"]

    # translator of id's, human to full
    matching = {}
    for i, x in enumerate(human):
        for j, y in enumerate(full):
            if np.all(x == y):
                matching[i] = j

    translated_human_edges = [
        (matching[u], matching[v])
        for u, v in human_edges
    ]

    # translator of id's half-full to full
    matching = {}
    for i, x in enumerate(half_full):
        for j, y in enumerate(full):
            if np.all(x == y):
                matching[i] = j

    translated_half_full_edges = [
        (matching[u], matching[v])
        for u, v in half_full_edges
    ]

    lattice = Lattice(
        nodes_positions=full,
        edges_indices=full_edges,
        linear_stiffness=10,
        angular_stiffness=0.2
    )

    if configuration == "full":
        for edge in lattice._possible_edges:
            lattice.flip_edge(edge)

    elif configuration == "human":
        for edge in lattice._possible_edges:
            edge_tuple = (edge._nodes[0].label, edge._nodes[1].label)
            if edge_tuple in translated_human_edges:
                if edge not in lattice.edges:
                    lattice.flip_edge(edge)

    elif configuration == "half_full":
        for edge in lattice._possible_edges:
            edge_tuple = (edge._nodes[0].label, edge._nodes[1].label)
            if edge_tuple in translated_half_full_edges:
                if edge not in lattice.edges:
                    lattice.flip_edge(edge)

    return lattice
