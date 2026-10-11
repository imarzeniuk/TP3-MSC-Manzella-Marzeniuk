import numpy


class Quad:
    """Cuadrilatero de referencia en coordenadas naturales (xi, eta), con xi y
    eta en [-1, 1].

    El orden de interpolacion se elige con el nombre:
        Quad("Q4"): bilineal, 4 nodos
        Quad("Q8"): cuadratico serendipito, 8 nodos

    Numeracion local: primero los vertices (antihorario) y, en el cuadratico,
    despues los nodos de mitad de lado (4 entre 0 y 1, 5 entre 1 y 2,
    6 entre 2 y 3, 7 entre 3 y 0):

              eta                           eta
               ^                             ^
        3 ----------- 2               3 ---- 6 ---- 2
        |      |      |               |      |      |
        |      +------|--> xi         7      +------5--> xi
        |             |               |             |
        0 ----------- 1               0 ---- 4 ---- 1
    """

    family = "quadrilateral"
    number_of_vertices = 4

    # Orden de interpolacion de cada nombre
    ORDERS = {"Q4": 1, "Q8": 2}

    def __init__(self, name):
        if name not in self.ORDERS:
            raise ValueError("Unknown quadrilateral: " + str(name))

        self.name = name
        self.order = self.ORDERS[name]

        if self.order == 1:
            self.number_of_nodes = 4

            # Una fila por nodo: [xi, eta]
            self.node_coordinates = numpy.array([
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ])

            # Nodos locales de cada lado
            self.edges = ((0, 1), (1, 2), (2, 3), (3, 0))

        else:
            self.number_of_nodes = 8

            self.node_coordinates = numpy.array([
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [0.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ])

            # Nodos locales de cada lado: los dos vertices y el nodo medio
            self.edges = ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7))

    def default_rule(self):
        """Regla de cuadratura de integracion completa (ver Quadrature):
        cantidad de puntos de Gauss por direccion.
        """
        if self.order == 1:
            return 2

        return 3

    def N(self, xi, eta):
        """Funciones de forma en (xi, eta). Devuelve un vector con una
        componente por nodo.

        Bilineal:
            N_i = (1 + xi_i xi) (1 + eta_i eta) / 4

        Cuadratico:
            vertices:        N_i = (1 + xi_i xi) (1 + eta_i eta) (xi_i xi + eta_i eta - 1) / 4
            lados xi_i = 0:  N_i = (1 - xi^2) (1 + eta_i eta) / 2
            lados eta_i = 0: N_i = (1 + xi_i xi) (1 - eta^2) / 2
        """
        if self.order == 1:
            return 0.25 * numpy.array([
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ])

        return numpy.array([
            0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0),
            0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0),
            0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0),
            0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0),
            0.5 * (1.0 - xi * xi) * (1.0 - eta),
            0.5 * (1.0 + xi) * (1.0 - eta * eta),
            0.5 * (1.0 - xi * xi) * (1.0 + eta),
            0.5 * (1.0 - xi) * (1.0 - eta * eta),
        ])

    def dN(self, xi, eta):
        """Derivadas naturales en (xi, eta). Devuelve una matriz de 2 x nodos:
        fila 0 = dN/dxi, fila 1 = dN/deta, una columna por nodo.
        """
        if self.order == 1:
            return 0.25 * numpy.array([
                [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
                [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
            ])

        return numpy.array([
            [
                0.25 * (1.0 - eta) * (2.0 * xi + eta),
                0.25 * (1.0 - eta) * (2.0 * xi - eta),
                0.25 * (1.0 + eta) * (2.0 * xi + eta),
                0.25 * (1.0 + eta) * (2.0 * xi - eta),
                -xi * (1.0 - eta),
                0.5 * (1.0 - eta * eta),
                -xi * (1.0 + eta),
                -0.5 * (1.0 - eta * eta),
            ],
            [
                0.25 * (1.0 - xi) * (xi + 2.0 * eta),
                0.25 * (1.0 + xi) * (-xi + 2.0 * eta),
                0.25 * (1.0 + xi) * (xi + 2.0 * eta),
                0.25 * (1.0 - xi) * (-xi + 2.0 * eta),
                -0.5 * (1.0 - xi * xi),
                -eta * (1.0 + xi),
                0.5 * (1.0 - xi * xi),
                -eta * (1.0 - xi),
            ],
        ])
