"""Verificaciones de las reglas de cuadratura.

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_quadrature
"""

import math

import numpy

from source.reference_elements.t3 import T3
from source.reference_elements.q4 import Q4
from source.reference_elements.t6 import T6
from source.reference_elements.q8 import Q8
from source.quadrature.quadrature import Quadrature


# Elemento de triangulo y grado total maximo que la regla integra exacto
TRIANGLES = [(T3(), 1), (T6(), 2)]

# Elemento cuadrilatero y grado maximo por variable que la regla integra exacto
QUADRILATERALS = [(Q4(), 3), (Q8(), 5)]


def integrate(quadrature, f):
    total = 0.0

    for g in range(quadrature.number_of_points):
        xi, eta = quadrature.points[g]
        total = total + quadrature.weights[g] * f(xi, eta)

    return total


def test_number_of_points():
    assert Quadrature(T3()).number_of_points == 1
    assert Quadrature(T6()).number_of_points == 3
    assert Quadrature(Q4()).number_of_points == 4
    assert Quadrature(Q8()).number_of_points == 9


def test_weights_sum_to_reference_area():
    for element, degree in TRIANGLES:
        assert numpy.isclose(numpy.sum(Quadrature(element).weights), 0.5)

    for element, degree in QUADRILATERALS:
        assert numpy.isclose(numpy.sum(Quadrature(element).weights), 4.0)


def test_triangle_monomials():
    # Integral exacta de xi^a eta^b en el triangulo: a! b! / (a + b + 2)!
    for element, degree in TRIANGLES:
        quadrature = Quadrature(element)

        for a in range(degree + 1):
            for b in range(degree + 1 - a):
                exact = (
                    math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)
                )
                value = integrate(quadrature, lambda xi, eta: xi**a * eta**b)

                assert numpy.isclose(value, exact)


def test_quadrilateral_monomials():
    # Integral exacta de x^n en [-1, 1]: 2 / (n + 1) si n es par, 0 si es impar
    def exact_1d(n):
        if n % 2 == 0:
            return 2.0 / (n + 1)

        return 0.0

    for element, degree in QUADRILATERALS:
        quadrature = Quadrature(element)

        for a in range(degree + 1):
            for b in range(degree + 1):
                exact = exact_1d(a) * exact_1d(b)
                value = integrate(quadrature, lambda xi, eta: xi**a * eta**b)

                assert numpy.isclose(value, exact)


def test_unknown_element():
    class Unknown:
        name = "H8"

    try:
        Quadrature(Unknown())
    except ValueError:
        return

    assert False


if __name__ == "__main__":
    tests = [
        test_number_of_points,
        test_weights_sum_to_reference_area,
        test_triangle_monomials,
        test_quadrilateral_monomials,
        test_unknown_element,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)
