"""Numeracion de los grados de libertad. Es la unica definicion del programa:
la malla (Mesh2D) y los elementos (PlaneElement) la usan desde aca.

Cada nodo tiene dos grados de libertad, u (en x) y v (en y). El nodo global n
tiene u en el DOF 2n y v en el DOF 2n + 1, asi que el vector global es
[u0, v0, u1, v1, ...].
"""

DOFS_PER_NODE = 2


def node_dofs(node_global_id):
    """DOFs globales de un nodo: [2n, 2n + 1]."""
    first = int(node_global_id) * DOFS_PER_NODE

    dofs = []

    for i in range(DOFS_PER_NODE):
        dofs.append(first + i)

    return dofs


def element_dofs(node_global_ids):
    """DOFs globales de un elemento, ordenados por nodo: [u0, v0, u1, v1, ...]."""
    dofs = []

    for node_global_id in node_global_ids:
        dofs.extend(node_dofs(node_global_id))

    return dofs
