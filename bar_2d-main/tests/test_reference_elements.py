"""Verificaciones de los elementos de referencia.

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_reference_elements
"""

import numpy

from source.reference_elements import make_reference_element
from source.reference_elements.tria import Tria
from source.reference_elements.quad import Quad
from source.reference_elements.line import Line


# Elementos planos
ELEMENTS = [Tria("CST"), Quad("Q4"), Tria("LST"), Quad("Q8")]

# Puntos naturales cualesquiera (validos para el triangulo y el cuadrilatero)
POINTS = [(0.0, 0.0), (0.2, 0.3), (0.6, 0.1), (0.25, 0.75), (1.0, 0.0)]

LINES = [Line("L2"), Line("L3")]

LINE_POINTS = [-1.0, -0.3, 0.0, 0.45, 1.0]


def raises_value_error(function, argument):
    try:
        function(argument)
    except ValueError:
        return True

    return False


def test_shapes():
    for element in ELEMENTS:
        N = element.N(0.2, 0.3)
        dN = element.dN(0.2, 0.3)

        assert N.shape == (element.number_of_nodes,)
        assert dN.shape == (2, element.number_of_nodes)
        assert element.node_coordinates.shape == (element.number_of_nodes, 2)


def test_partition_of_unity():
    # La suma de las N vale 1 en cualquier punto
    for element in ELEMENTS:
        for xi, eta in POINTS:
            assert numpy.isclose(numpy.sum(element.N(xi, eta)), 1.0)


def test_derivatives_sum_to_zero():
    # Consecuencia de lo anterior: la suma de las derivadas vale 0
    for element in ELEMENTS:
        for xi, eta in POINTS:
            assert numpy.allclose(numpy.sum(element.dN(xi, eta), axis=1), 0.0)


def test_kronecker_delta():
    # N_i vale 1 en el nodo i y 0 en los demas
    for element in ELEMENTS:
        n = element.number_of_nodes

        for i in range(n):
            xi, eta = element.node_coordinates[i]
            assert numpy.allclose(element.N(xi, eta), numpy.eye(n)[i])


def test_derivatives_against_finite_differences():
    # dN coincide con la derivada numerica (diferencias centradas) de N
    h = 1.0e-6

    for element in ELEMENTS:
        for xi, eta in POINTS:
            dN_dxi = (element.N(xi + h, eta) - element.N(xi - h, eta)) / (2.0 * h)
            dN_deta = (element.N(xi, eta + h) - element.N(xi, eta - h)) / (2.0 * h)

            dN = element.dN(xi, eta)

            assert numpy.allclose(dN[0], dN_dxi, atol=1.0e-8)
            assert numpy.allclose(dN[1], dN_deta, atol=1.0e-8)


def test_reference_jacobian():
    # Con los nodos fisicos iguales a los naturales, J = dN @ coords = identidad
    for element in ELEMENTS:
        for xi, eta in POINTS:
            J = element.dN(xi, eta) @ element.node_coordinates
            assert numpy.allclose(J, numpy.eye(2))


def test_names_orders_and_families():
    # (elemento, familia, orden, nodos, vertices, regla completa)
    expected = [
        (Tria("CST"), "triangle", 1, 3, 3, "1pt"),
        (Tria("LST"), "triangle", 2, 6, 3, "3pt"),
        (Quad("Q4"), "quadrilateral", 1, 4, 4, 2),
        (Quad("Q8"), "quadrilateral", 2, 8, 4, 3),
        (Line("L2"), "line", 1, 2, 2, 2),
        (Line("L3"), "line", 2, 3, 2, 3),
    ]

    for element, family, order, nodes, vertices, rule in expected:
        assert element.family == family
        assert element.order == order
        assert element.number_of_nodes == nodes
        assert element.number_of_vertices == vertices
        assert element.default_rule() == rule


def test_unknown_names_are_rejected():
    assert raises_value_error(Tria, "Q4")
    assert raises_value_error(Tria, "T10")
    assert raises_value_error(Quad, "CST")
    assert raises_value_error(Quad, "Q9")
    assert raises_value_error(Line, "L4")
    assert raises_value_error(make_reference_element, "H8")


def test_factory_and_aliases():
    # (nombre, clase, nombre del elemento creado, orden)
    expected = [
        ("CST", Tria, "CST", 1),
        ("LST", Tria, "LST", 2),
        ("Q4", Quad, "Q4", 1),
        ("Q8", Quad, "Q8", 2),
        ("L2", Line, "L2", 1),
        ("L3", Line, "L3", 2),
        ("T3", Tria, "CST", 1),
        ("T6", Tria, "LST", 2),
    ]

    for name, element_class, element_name, order in expected:
        element = make_reference_element(name)

        # isinstance(objeto, Clase): True si el objeto es de esa clase
        assert isinstance(element, element_class)
        assert element.name == element_name
        assert element.order == order


def test_edges():
    for element in ELEMENTS:
        assert len(element.edges) == element.number_of_vertices

        for k in range(len(element.edges)):
            edge = element.edges[k]

            # Cada lado va de un vertice al siguiente, en sentido antihorario
            assert edge[0] == k
            assert edge[1] == (k + 1) % element.number_of_vertices
            assert len(edge) == element.order + 1

            # En los cuadraticos el tercer nodo esta en el punto medio del lado
            if element.order == 2:
                a = element.node_coordinates[edge[0]]
                b = element.node_coordinates[edge[1]]

                assert numpy.allclose(element.node_coordinates[edge[2]], 0.5 * (a + b))


def test_edge_restriction_matches_line_element():
    # Sobre un lado, las N de los nodos de ese lado coinciden con las del
    # elemento de linea del mismo orden, y las de los demas nodos valen 0
    for element in ELEMENTS:
        line = make_reference_element("L2" if element.order == 1 else "L3")

        for edge in element.edges:
            a = element.node_coordinates[edge[0]]
            b = element.node_coordinates[edge[1]]

            for s in LINE_POINTS:
                xi, eta = 0.5 * (1.0 - s) * a + 0.5 * (1.0 + s) * b
                N = element.N(xi, eta)

                expected = numpy.zeros(element.number_of_nodes)
                expected[list(edge)] = line.N(s)

                assert numpy.allclose(N, expected)


def test_line_elements():
    h = 1.0e-6

    for line in LINES:
        n = line.number_of_nodes

        assert line.node_coordinates.shape == (n, 1)

        for i in range(n):
            assert numpy.allclose(line.N(line.node_coordinates[i, 0]), numpy.eye(n)[i])

        for xi in LINE_POINTS:
            N = line.N(xi)
            dN = line.dN(xi)

            assert N.shape == (n,)
            assert dN.shape == (1, n)
            assert numpy.isclose(numpy.sum(N), 1.0)
            assert numpy.isclose(numpy.sum(dN), 0.0)

            dN_dxi = (line.N(xi + h) - line.N(xi - h)) / (2.0 * h)

            assert numpy.allclose(dN[0], dN_dxi, atol=1.0e-8)


if __name__ == "__main__":
    tests = [
        test_shapes,
        test_partition_of_unity,
        test_derivatives_sum_to_zero,
        test_kronecker_delta,
        test_derivatives_against_finite_differences,
        test_reference_jacobian,
        test_names_orders_and_families,
        test_unknown_names_are_rejected,
        test_factory_and_aliases,
        test_edges,
        test_edge_restriction_matches_line_element,
        test_line_elements,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)
