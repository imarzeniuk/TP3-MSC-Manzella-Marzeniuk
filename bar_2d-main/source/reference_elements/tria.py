import numpy


class Tria:
    """Triangulo de referencia en coordenadas naturales (xi, eta).

    El orden de interpolacion se elige con el nombre:
        Tria("CST"): lineal, 3 nodos
        Tria("LST"): cuadratico, 6 nodos

    Numeracion local: primero los vertices (antihorario) y, en el cuadratico,
    despues los nodos de mitad de lado (3 entre 0 y 1, 4 entre 1 y 2, 5 entre 2 y 0):

        eta                       eta
         ^                         ^
         2                         2
         | \\                       | \\
         |   \\                     5   4
         |     \\                   |     \\
         0 ----- 1 --> xi          0 - 3 - 1 --> xi
    """

    family = "triangle"
    number_of_vertices = 3

    # Orden de interpolacion de cada nombre
    ORDERS = {"CST": 1, "LST": 2}

    def __init__(self, name):
        if name not in self.ORDERS:
            raise ValueError("Unknown triangle: " + str(name))

        self.name = name
        self.order = self.ORDERS[name]

        if self.order == 1:
            self.number_of_nodes = 3

            # Una fila por nodo: [xi, eta]
            self.node_coordinates = numpy.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ])

            # Nodos locales de cada lado
            self.edges = ((0, 1), (1, 2), (2, 0))

        else:
            self.number_of_nodes = 6

            self.node_coordinates = numpy.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.0],
                [0.5, 0.5],
                [0.0, 0.5],
            ])

            # Nodos locales de cada lado: los dos vertices y el nodo medio
            self.edges = ((0, 1, 3), (1, 2, 4), (2, 0, 5))

    def default_rule(self):
        """Regla de cuadratura de integracion completa (ver Quadrature)."""
        if self.order == 1:
            return "1pt"

        return "3pt"

    def N(self, xi, eta):
        """Funciones de forma en (xi, eta). Devuelve un vector con una
        componente por nodo.

        Con las coordenadas de area L1 = 1 - xi - eta, L2 = xi, L3 = eta:
            lineal:     N = L
            cuadratico: vertices N = L (2 L - 1), mitad de lado N = 4 La Lb
        """
        L1 = 1.0 - xi - eta

        if self.order == 1:
            return numpy.array([L1, xi, eta])

        return numpy.array([
            L1 * (2.0 * L1 - 1.0),
            xi * (2.0 * xi - 1.0),
            eta * (2.0 * eta - 1.0),
            4.0 * L1 * xi,
            4.0 * xi * eta,
            4.0 * eta * L1,
        ])

    def dN(self, xi, eta):
        """Derivadas naturales en (xi, eta). Devuelve una matriz de 2 x nodos:
        fila 0 = dN/dxi, fila 1 = dN/deta, una columna por nodo.

        En el lineal son constantes (no dependen del punto).
        """
        if self.order == 1:
            return numpy.array([
                [-1.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0],
            ])

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
