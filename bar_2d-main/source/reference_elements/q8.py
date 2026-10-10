import numpy


class Q8:
    """Cuadrilatero cuadratico de 8 nodos (serendipito) en coordenadas
    naturales (xi, eta), con xi y eta en [-1, 1].

    Numeracion local: primero los vertices (antihorario) y despues los nodos
    de mitad de lado (4 entre 0 y 1, 5 entre 1 y 2, 6 entre 2 y 3, 7 entre 3 y 0):

              eta
               ^
        3 ---- 6 ---- 2
        |      |      |
        7      +------5--> xi
        |             |
        0 ---- 4 ---- 1
    """

    name = "Q8"
    number_of_nodes = 8

    # Una fila por nodo: [xi, eta]
    node_coordinates = numpy.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    def N(self, xi, eta):
        """Funciones de forma en (xi, eta). Devuelve un vector de 8 componentes.

        Vertices:        N_i = (1 + xi_i xi) (1 + eta_i eta) (xi_i xi + eta_i eta - 1) / 4
        Lados xi_i = 0:  N_i = (1 - xi^2) (1 + eta_i eta) / 2
        Lados eta_i = 0: N_i = (1 + xi_i xi) (1 - eta^2) / 2
        """
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
        """Derivadas naturales en (xi, eta). Devuelve una matriz de 2 x 8:
        fila 0 = dN/dxi, fila 1 = dN/deta, una columna por nodo.
        """
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
