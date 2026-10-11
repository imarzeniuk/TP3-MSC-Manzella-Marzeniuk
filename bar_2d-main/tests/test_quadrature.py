"""Verificaciones de las reglas de cuadratura.

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_quadrature
"""

import math

import numpy

from source.reference_elements.tria import Tria
from source.reference_elements.quad import Quad
from source.reference_elements.line import Line
from source.quadrature.quadrature import Quadrature


# Reglas de triangulo a verificar
TRIANGLE_RULES = ["1pt", "3pt"] + [("collapsed", n) for n in range(1, 6)]


def integrate(quadrature, f):
    # f recibe el punto como una fila: [xi, eta] (o [xi] en la linea)
    total = 0.0

    for g in range(quadrature.number_of_points):
        total = total + quadrature.weights[g] * f(quadrature.points[g])

    return total


def exact_1d(n):
    # Integral exacta de x^n en [-1, 1]: 2 / (n + 1) si n es par, 0 si es impar
    if n % 2 == 0:
        return 2.0 / (n + 1)

    return 0.0


def raises_value_error(family, rule):
    try:
        Quadrature(family, rule)
    except ValueError:
        return True

    return False


def test_number_of_points_and_degree():
    for n in range(1, 6):
        assert Quadrature("line", n).number_of_points == n
        assert Quadrature("line", n).degree == 2 * n - 1
        assert Quadrature("line", n).points.shape == (n, 1)

        assert Quadrature("quadrilateral", n).number_of_points == n * n
        assert Quadrature("quadrilateral", n).degree == 2 * n - 1
        assert Quadrature("quadrilateral", n).points.shape == (n * n, 2)

        assert Quadrature("triangle", ("collapsed", n)).number_of_points == n * n
        assert Quadrature("triangle", ("collapsed", n)).degree == 2 * n - 2

    assert Quadrature("triangle", "1pt").number_of_points == 1
    assert Quadrature("triangle", "1pt").degree == 1
    assert Quadrature("triangle", "3pt").number_of_points == 3
    assert Quadrature("triangle", "3pt").degree == 2


def test_weights_sum_to_reference_size():
    for n in range(1, 6):
        assert numpy.isclose(numpy.sum(Quadrature("line", n).weights), 2.0)
        assert numpy.isclose(numpy.sum(Quadrature("quadrilateral", n).weights), 4.0)

    for rule in TRIANGLE_RULES:
        assert numpy.isclose(numpy.sum(Quadrature("triangle", rule).weights), 0.5)


def test_points_are_inside_the_triangle():
    for rule in TRIANGLE_RULES:
        for xi, eta in Quadrature("triangle", rule).points:
            assert xi > 0.0 and eta > 0.0 and xi + eta < 1.0


def test_line_monomials():
    # Gauss-Legendre de n puntos integra exacto hasta grado 2n - 1
    for n in range(1, 6):
        quadrature = Quadrature("line", n)

        for a in range(quadrature.degree + 1):
            value = integrate(quadrature, lambda p: p[0]**a)

            assert numpy.isclose(value, exact_1d(a))

        # ... y no integra exacto el grado siguiente (2n)
        value = integrate(quadrature, lambda p: p[0]**(2 * n))

        assert not numpy.isclose(value, exact_1d(2 * n))


def test_quadrilateral_monomials():
    # Gauss n x n integra exacto hasta grado 2n - 1 en cada variable
    for n in range(1, 6):
        quadrature = Quadrature("quadrilateral", n)

        for a in range(quadrature.degree + 1):
            for b in range(quadrature.degree + 1):
                value = integrate(quadrature, lambda p: p[0]**a * p[1]**b)

                assert numpy.isclose(value, exact_1d(a) * exact_1d(b))


def test_quadrilateral_point_order():
    # xi varia mas rapido que eta
    points = Quadrature("quadrilateral", 2).points
    a = 1.0 / math.sqrt(3.0)

    assert numpy.allclose(points, [[-a, -a], [a, -a], [-a, a], [a, a]])


def test_triangle_monomials():
    # Integral exacta de xi^a eta^b en el triangulo: a! b! / (a + b + 2)!
    for rule in TRIANGLE_RULES:
        quadrature = Quadrature("triangle", rule)

        for a in range(quadrature.degree + 1):
            for b in range(quadrature.degree + 1 - a):
                exact = (
                    math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)
                )
                value = integrate(quadrature, lambda p: p[0]**a * p[1]**b)

                assert numpy.isclose(value, exact)


def test_for_element_uses_full_integration():
    # (elemento, familia, cantidad de puntos de la regla completa)
    expected = [
        (Tria("CST"), "triangle", 1),
        (Tria("LST"), "triangle", 3),
        (Quad("Q4"), "quadrilateral", 4),
        (Quad("Q8"), "quadrilateral", 9),
        (Line("L2"), "line", 2),
        (Line("L3"), "line", 3),
    ]

    for element, family, number_of_points in expected:
        quadrature = Quadrature.for_element(element)

        assert quadrature.family == family
        assert quadrature.number_of_points == number_of_points


def test_for_element_with_explicit_rule():
    # La regla no depende del elemento: Q8 con Gauss 2x2 (reducida)
    assert Quadrature.for_element(Quad("Q8"), 2).number_of_points == 4
    assert Quadrature.for_element(Tria("CST"), "3pt").number_of_points == 3


def test_invalid_rules_are_rejected():
    assert raises_value_error("hexahedron", 2)
    assert raises_value_error("line", 0)
    assert raises_value_error("line", 2.0)
    assert raises_value_error("quadrilateral", "3pt")
    assert raises_value_error("quadrilateral", -1)
    assert raises_value_error("triangle", 2)
    assert raises_value_error("triangle", "7pt")
    assert raises_value_error("triangle", ("collapsed", 0))
    assert raises_value_error("line", True)


def test_numpy_integers_are_accepted():
    # Un numero de puntos leido de un arreglo de NumPy es numpy.int64, no int
    assert Quadrature("quadrilateral", numpy.int64(3)).number_of_points == 9
    assert Quadrature("line", numpy.int32(2)).number_of_points == 2
    assert Quadrature("triangle", ("collapsed", numpy.int64(2))).number_of_points == 4


if __name__ == "__main__":
    tests = [
        test_number_of_points_and_degree,
        test_weights_sum_to_reference_size,
        test_points_are_inside_the_triangle,
        test_line_monomials,
        test_quadrilateral_monomials,
        test_quadrilateral_point_order,
        test_triangle_monomials,
        test_for_element_uses_full_integration,
        test_for_element_with_explicit_rule,
        test_invalid_rules_are_rejected,
        test_numpy_integers_are_accepted,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)
