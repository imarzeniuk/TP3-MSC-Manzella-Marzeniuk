"""Verificaciones de la matriz de rigidez local (PlaneElement).

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_plane_element
"""

import numpy

from source.reference_elements.t3 import T3
from source.elements.element_set import ElementSet
from source.elements.plane_element import PlaneElement
from tests.test_element_set import CASES, polygon_area


E = 21.0e9
nu = 0.25
thickness = 0.5

# Tension plana
D = E / (1.0 - nu * nu) * numpy.array([
    [1.0, nu, 0.0],
    [nu, 1.0, 0.0],
    [0.0, 0.0, (1.0 - nu) / 2.0],
])


def all_elements():
    # Lista de (elemento, vertices) para los dos elementos de cada tipo
    elements = []

    for element_set, corners_list in CASES:
        for e in range(element_set.number_of_elements):
            elements.append(
                (PlaneElement(element_set, e, D, thickness), corners_list[e])
            )

    return elements


def nodal_displacements(element, field):
    # Evalua field(x, y) -> (u, v) en los nodos: devuelve [u0, v0, u1, v1, ...]
    ue = []

    for node_global_id in element.node_global_ids:
        x, y = element.element_set.node_coordinates[node_global_id]
        u, v = field(x, y)
        ue.append(u)
        ue.append(v)

    return numpy.array(ue)


def test_size_and_symmetry():
    for element, corners in all_elements():
        Ke = element.stiffness_matrix()
        n = 2 * element.element_set.nodes_per_element

        assert Ke.shape == (n, n)
        assert numpy.allclose(Ke, Ke.transpose(), rtol=1.0e-12, atol=1.0e-3)


def test_rigid_body_motions_give_no_forces():
    fields = [
        lambda x, y: (1.0, 0.0),  # traslacion en x
        lambda x, y: (0.0, 1.0),  # traslacion en y
        lambda x, y: (-y, x),     # rotacion (pequena) alrededor del origen
    ]

    for element, corners in all_elements():
        Ke = element.stiffness_matrix()

        for field in fields:
            forces = Ke @ nodal_displacements(element, field)

            # Tolerancia relativa al tamano de los terminos de Ke
            assert numpy.allclose(forces, 0.0, atol=1.0e-9 * numpy.max(numpy.abs(Ke)))


def test_only_three_zero_energy_modes():
    # Con integracion completa los unicos modos sin energia son los 3 rigidos
    for element, corners in all_elements():
        Ke = element.stiffness_matrix()
        eigenvalues = numpy.linalg.eigvalsh(Ke)

        number_of_zeros = numpy.sum(eigenvalues < 1.0e-9 * numpy.max(eigenvalues))

        assert number_of_zeros == 3


def test_constant_strain_energy():
    # Para u = a x + c y, v = b y: eps = [a, b, c] constante y
    # u^T Ke u = eps^T D eps * area * espesor
    a, b, c = 1.0e-3, -2.0e-3, 0.5e-3
    eps = numpy.array([a, b, c])

    for element, corners in all_elements():
        Ke = element.stiffness_matrix()
        ue = nodal_displacements(element, lambda x, y: (a * x + c * y, b * y))

        energy = ue @ Ke @ ue
        exact = eps @ D @ eps * polygon_area(corners) * thickness

        assert numpy.isclose(energy, exact)


def test_B_matrix_gives_constant_strain():
    a, b, c = 1.0e-3, -2.0e-3, 0.5e-3

    for element, corners in all_elements():
        ue = nodal_displacements(element, lambda x, y: (a * x + c * y, b * y))

        for g in range(element.element_set.number_of_points):
            assert numpy.allclose(element.B_matrix(g) @ ue, [a, b, c])


def test_global_dofs():
    node_coordinates = numpy.zeros((6, 2))
    node_coordinates[2] = [0.0, 0.0]
    node_coordinates[0] = [1.0, 0.0]
    node_coordinates[5] = [0.0, 1.0]

    element_set = ElementSet(T3(), node_coordinates, [[2, 0, 5]])
    element = PlaneElement(element_set, 0, D, thickness)

    assert element.global_dofs() == [4, 5, 0, 1, 10, 11]


if __name__ == "__main__":
    tests = [
        test_size_and_symmetry,
        test_rigid_body_motions_give_no_forces,
        test_only_three_zero_energy_modes,
        test_constant_strain_energy,
        test_B_matrix_gives_constant_strain,
        test_global_dofs,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)
