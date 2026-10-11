"""Verificaciones de ElementSet.

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_element_set
"""

import numpy

from source.reference_elements.t3 import T3
from source.reference_elements.q4 import Q4
from source.reference_elements.t6 import T6
from source.reference_elements.q8 import Q8
from source.elements.element_set import ElementSet


# Vertices [x, y] de dos elementos distorsionados de cada forma (antihorarios)
TRIANGLE_CORNERS = [
    [[0.0, 0.0], [2.0, 0.5], [0.5, 1.5]],
    [[3.0, 1.0], [5.0, 1.0], [3.5, 4.0]],
]

QUADRILATERAL_CORNERS = [
    [[0.0, 0.0], [2.0, 0.2], [2.5, 1.8], [-0.3, 1.5]],
    [[4.0, 1.0], [7.0, 0.0], [6.0, 3.0], [4.5, 2.0]],
]


def build_element_set(reference_element, linear_element, corners_list):
    # Los nodos fisicos se obtienen mapeando los nodos naturales del elemento
    # con las funciones de forma lineales: asi los lados quedan rectos y los
    # nodos de mitad de lado caen en el punto medio.
    node_coordinates = []
    connectivity = []

    for corners in corners_list:
        element_nodes = []

        for xi, eta in reference_element.node_coordinates:
            element_nodes.append(len(node_coordinates))
            node_coordinates.append(linear_element.N(xi, eta) @ numpy.array(corners))

        connectivity.append(element_nodes)

    return ElementSet(reference_element, node_coordinates, connectivity)


def polygon_area(corners):
    area = 0.0

    for i in range(len(corners)):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % len(corners)]
        area = area + 0.5 * (x1 * y2 - x2 * y1)

    return area


# Cada caso: (conjunto de elementos, vertices de cada elemento)
CASES = [
    (build_element_set(T3(), T3(), TRIANGLE_CORNERS), TRIANGLE_CORNERS),
    (build_element_set(T6(), T3(), TRIANGLE_CORNERS), TRIANGLE_CORNERS),
    (build_element_set(Q4(), Q4(), QUADRILATERAL_CORNERS), QUADRILATERAL_CORNERS),
    (build_element_set(Q8(), Q4(), QUADRILATERAL_CORNERS), QUADRILATERAL_CORNERS),
]


def test_shape_matches_reference_element():
    for element_set, corners_list in CASES:
        for g in range(element_set.number_of_points):
            xi, eta = element_set.quadrature.points[g]
            N = element_set.reference_element.N(xi, eta)

            for a in range(element_set.nodes_per_element):
                assert numpy.isclose(element_set.shape(g, a), N[a])


def test_wdetJ_sums_to_element_area():
    for element_set, corners_list in CASES:
        for e in range(element_set.number_of_elements):
            area = 0.0

            for g in range(element_set.number_of_points):
                area = area + element_set.wdetJ(e, g)

            assert numpy.isclose(area, polygon_area(corners_list[e]))


def test_dshape_sums_to_zero():
    for element_set, corners_list in CASES:
        for e in range(element_set.number_of_elements):
            for g in range(element_set.number_of_points):
                for variable in range(2):
                    total = 0.0

                    for a in range(element_set.nodes_per_element):
                        total = total + element_set.dshape(e, g, a, variable)

                    assert numpy.isclose(total, 0.0)


def test_gradient_of_linear_field():
    # Para u = 1 + 2 x + 3 y el gradiente interpolado es (2, 3) en todo punto
    for element_set, corners_list in CASES:
        for e in range(element_set.number_of_elements):
            for g in range(element_set.number_of_points):
                du_dx = 0.0
                du_dy = 0.0

                for a in range(element_set.nodes_per_element):
                    x, y = element_set.node_coordinates[element_set.connectivity[e][a]]
                    u = 1.0 + 2.0 * x + 3.0 * y

                    du_dx = du_dx + element_set.dshape(e, g, a, 0) * u
                    du_dy = du_dy + element_set.dshape(e, g, a, 1) * u

                assert numpy.isclose(du_dx, 2.0)
                assert numpy.isclose(du_dy, 3.0)


def test_clockwise_element_is_rejected():
    node_coordinates = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    try:
        ElementSet(T3(), node_coordinates, [[0, 2, 1]])
    except ValueError:
        return

    assert False


def test_wrong_number_of_nodes_is_rejected():
    node_coordinates = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    try:
        ElementSet(Q4(), node_coordinates, [[0, 1, 2]])
    except ValueError:
        return

    assert False


if __name__ == "__main__":
    tests = [
        test_shape_matches_reference_element,
        test_wdetJ_sums_to_element_area,
        test_dshape_sums_to_zero,
        test_gradient_of_linear_field,
        test_clockwise_element_is_rejected,
        test_wrong_number_of_nodes_is_rejected,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)
