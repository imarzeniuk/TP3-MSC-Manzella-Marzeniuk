import numpy


class Q4:
    """Cuadrilatero bilineal de 4 nodos en coordenadas naturales (xi, eta),
    con xi y eta en [-1, 1].

    Numeracion local (antihoraria):

              eta
               ^
        3 ----------- 2
        |      |      |
        |      +------|--> xi
        |             |
        0 ----------- 1
    """

    name = "Q4"
    number_of_nodes = 4

    # Una fila por nodo: [xi, eta]
    node_coordinates = numpy.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ])

    def N(self, xi, eta):
        """Funciones de forma en (xi, eta). Devuelve un vector de 4 componentes.

        N_i = (1 + xi_i * xi) * (1 + eta_i * eta) / 4
        """
        return 0.25 * numpy.array([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ])

    def dN(self, xi, eta):
        """Derivadas naturales en (xi, eta). Devuelve una matriz de 2 x 4:
        fila 0 = dN/dxi, fila 1 = dN/deta, una columna por nodo.
        """
        return 0.25 * numpy.array([
            [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        ])
