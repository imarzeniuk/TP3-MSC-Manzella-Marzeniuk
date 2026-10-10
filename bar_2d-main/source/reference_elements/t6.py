import numpy


class T6:
    """Triangulo cuadratico de 6 nodos (LST) en coordenadas naturales (xi, eta).

    Numeracion local: primero los vertices (antihorario) y despues los nodos
    de mitad de lado (3 entre 0 y 1, 4 entre 1 y 2, 5 entre 2 y 0):

        eta
         ^
         2
         | \\
         5   4
         |     \\
         0 - 3 - 1 --> xi
    """

    name = "T6"
    number_of_nodes = 6

    # Una fila por nodo: [xi, eta]
    node_coordinates = numpy.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
    ])

    def N(self, xi, eta):
        """Funciones de forma en (xi, eta). Devuelve un vector de 6 componentes.

        Con las coordenadas de area L1 = 1 - xi - eta, L2 = xi, L3 = eta:
        vertices N = L (2 L - 1), mitad de lado N = 4 La Lb.
        """
        L1 = 1.0 - xi - eta

        return numpy.array([
            L1 * (2.0 * L1 - 1.0),
            xi * (2.0 * xi - 1.0),
            eta * (2.0 * eta - 1.0),
            4.0 * L1 * xi,
            4.0 * xi * eta,
            4.0 * eta * L1,
        ])

    def dN(self, xi, eta):
        """Derivadas naturales en (xi, eta). Devuelve una matriz de 2 x 6:
        fila 0 = dN/dxi, fila 1 = dN/deta, una columna por nodo.
        """
        L1 = 1.0 - xi - eta

        return numpy.array([
            [
                1.0 - 4.0 * L1,
                4.0 * xi - 1.0,
                0.0,
                4.0 * (L1 - xi),
                4.0 * eta,
                -4.0 * eta,
            ],
            [
                1.0 - 4.0 * L1,
                0.0,
                4.0 * eta - 1.0,
                -4.0 * xi,
                4.0 * xi,
                4.0 * (L1 - eta),
            ],
        ])
