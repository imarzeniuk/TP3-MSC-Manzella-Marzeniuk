"""Verificaciones de los elementos de referencia.

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_reference_elements
"""

import numpy

from source.reference_elements.t3 import T3
from source.reference_elements.q4 import Q4
from source.reference_elements.t6 import T6
from source.reference_elements.q8 import Q8


ELEMENTS = [T3(), Q4(), T6(), Q8()]

# Puntos naturales cualesquiera (validos para el triangulo y el cuadrilatero)
POINTS = [(0.0, 0.0), (0.2, 0.3), (0.6, 0.1), (0.25, 0.75), (1.0, 0.0)]


def test_shapes():
    for element in ELEMENTS:
        N = element.N(0.2, 0.3)
        dN = element.dN(0.2, 0.3)

        assert N.shape == (element.number_of_nodes,)
        assert dN.shape == (2, element.number_of_nodes)


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


if __name__ == "__main__":
    tests = [
        test_shapes,
        test_partition_of_unity,
        test_derivatives_sum_to_zero,
        test_kronecker_delta,
        test_derivatives_against_finite_differences,
        test_reference_jacobian,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)
