import numbers

import numpy


class Quadrature:
    """Regla de cuadratura sobre un dominio de referencia.

    La integral se aproxima como

        integral f dA  ~  sum_g  weights[g] * f(points[g])

    La regla depende de la familia geometrica y no del elemento: cualquier
    regla de una familia sirve para cualquier elemento de esa familia (por
    ejemplo, un Q8 se puede integrar con Gauss 3x3 o con 2x2 reducida).

    family = "line" (xi en [-1, 1]):
        rule = n (entero): Gauss-Legendre de n puntos.
        Exacta hasta grado 2n - 1. Los pesos suman 2.

    family = "quadrilateral" (xi, eta en [-1, 1]):
        rule = n (entero): Gauss-Legendre de n x n puntos.
        Exacta hasta grado 2n - 1 en cada variable. Los pesos suman 4.

    family = "triangle" (vertices (0,0), (1,0), (0,1)). Los pesos suman 1/2:
        rule = "1pt": 1 punto en el baricentro, exacta hasta grado 1.
        rule = "3pt": 3 puntos interiores, exacta hasta grado 2.
        rule = ("collapsed", n): Gauss-Legendre de n x n puntos llevada del
            cuadrado al triangulo, exacta hasta grado 2n - 2.

    Atributos:
        points:           una fila por punto, [xi, eta] (o [xi] en la linea)
        weights:          un peso por punto
        number_of_points: cantidad de puntos
        degree:           grado polinomial que la regla integra exacto
    """

    def __init__(self, family, rule):
        self.family = family
        self.rule = rule

        if family == "line":
            n = self.check_number_of_points(rule)
            points_1d, weights_1d = numpy.polynomial.legendre.leggauss(n)

            points = [[x] for x in points_1d]
            weights = weights_1d
            degree = 2 * n - 1

        elif family == "quadrilateral":
            n = self.check_number_of_points(rule)
            points_1d, weights_1d = numpy.polynomial.legendre.leggauss(n)

            points = []
            weights = []

            # Producto tensorial: xi varia mas rapido que eta
            for j in range(n):
                for i in range(n):
                    points.append([points_1d[i], points_1d[j]])
                    weights.append(weights_1d[i] * weights_1d[j])

            degree = 2 * n - 1

        elif family == "triangle":
            if rule == "1pt":
                points = [[1.0 / 3.0, 1.0 / 3.0]]
                weights = [0.5]
                degree = 1

            elif rule == "3pt":
                points = [
                    [1.0 / 6.0, 1.0 / 6.0],
                    [2.0 / 3.0, 1.0 / 6.0],
                    [1.0 / 6.0, 2.0 / 3.0],
                ]
                weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
                degree = 2

            elif isinstance(rule, tuple) and len(rule) == 2 and rule[0] == "collapsed":
                n = self.check_number_of_points(rule[1])
                points_1d, weights_1d = numpy.polynomial.legendre.leggauss(n)

                points = []
                weights = []

                # El cuadrado (a, b) se "colapsa" al triangulo:
                # xi = (1 + a)(1 - b) / 4, eta = (1 + b) / 2, con jacobiano (1 - b) / 8
                for j in range(n):
                    for i in range(n):
                        a = points_1d[i]
                        b = points_1d[j]

                        points.append([(1.0 + a) * (1.0 - b) / 4.0, (1.0 + b) / 2.0])
                        weights.append(weights_1d[i] * weights_1d[j] * (1.0 - b) / 8.0)

                degree = 2 * n - 2

            else:
                raise ValueError("Unknown triangle rule: " + str(rule))

        else:
            raise ValueError("Unknown family: " + str(family))

        self.points = numpy.array(points)
        self.weights = numpy.array(weights)
        self.number_of_points = len(self.weights)
        self.degree = degree

    @classmethod
    def for_element(cls, reference_element, rule=None):
        """Regla para un elemento de referencia. Si no se indica rule, se usa
        la de integracion completa del elemento (reference_element.default_rule()).
        """
        if rule is None:
            rule = reference_element.default_rule()

        return cls(reference_element.family, rule)

    def check_number_of_points(self, n):
        # numbers.Integral acepta int de Python y enteros de NumPy (numpy.int64).
        # bool tambien es un tipo de entero en Python: se descarta para no aceptar True
        if not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 1:
            raise ValueError("Expected a positive integer number of points, got " + str(n))

        return int(n)
