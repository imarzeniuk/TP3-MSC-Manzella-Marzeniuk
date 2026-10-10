import math

import numpy


class Quadrature:
    """Regla de cuadratura (integracion completa) para un elemento de referencia.

    La integral sobre el elemento de referencia se aproxima como

        integral f(xi, eta) dxi deta  ~  sum_g  weights[g] * f(points[g])

    Reglas usadas:
        T3: 1 punto   (exacta hasta grado 1)
        T6: 3 puntos  (exacta hasta grado 2)
        Q4: Gauss 2x2 (exacta hasta grado 3 en xi y en eta)
        Q8: Gauss 3x3 (exacta hasta grado 5 en xi y en eta)

    En los triangulos los pesos suman 1/2 y en los cuadrilateros suman 4,
    que son las areas de los elementos de referencia.
    """

    def __init__(self, reference_element):
        name = reference_element.name

        if name == "T3":
            points = [[1.0 / 3.0, 1.0 / 3.0]]
            weights = [0.5]

        elif name == "T6":
            points = [
                [1.0 / 6.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0],
            ]
            weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]

        elif name == "Q4":
            a = 1.0 / math.sqrt(3.0)

            points, weights = self.tensor_product([-a, a], [1.0, 1.0])

        elif name == "Q8":
            a = math.sqrt(3.0 / 5.0)

            points, weights = self.tensor_product(
                [-a, 0.0, a],
                [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
            )

        else:
            raise ValueError("Unknown reference element: " + str(name))

        # Una fila por punto de Gauss: [xi, eta]
        self.points = numpy.array(points)
        self.weights = numpy.array(weights)
        self.number_of_points = len(weights)

    def tensor_product(self, points_1d, weights_1d):
        """Arma la regla 2D de un cuadrilatero combinando una regla de Gauss 1D
        en xi con la misma regla en eta. El peso 2D es el producto de los pesos.
        """
        points = []
        weights = []

        for j in range(len(points_1d)):
            for i in range(len(points_1d)):
                points.append([points_1d[i], points_1d[j]])
                weights.append(weights_1d[i] * weights_1d[j])

        return points, weights
