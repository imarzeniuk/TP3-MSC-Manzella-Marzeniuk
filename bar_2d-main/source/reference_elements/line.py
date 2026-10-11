import numpy


class Line:
    """Linea de referencia en la coordenada natural xi, con xi en [-1, 1].
    Sirve para integrar cargas sobre los lados de los elementos planos.

    El orden de interpolacion se elige con el nombre:
        Line("L2"): lineal, 2 nodos
        Line("L3"): cuadratico, 3 nodos

    Numeracion local: primero los extremos y despues el nodo medio, igual que
    en los lados (edges) de Tria y Quad:

        0 ----------- 1 --> xi        0 ---- 2 ---- 1 --> xi
    """

    family = "line"
    number_of_vertices = 2

    # Orden de interpolacion de cada nombre
    ORDERS = {"L2": 1, "L3": 2}

    def __init__(self, name):
        if name not in self.ORDERS:
            raise ValueError("Unknown line: " + str(name))

        self.name = name
        self.order = self.ORDERS[name]

        if self.order == 1:
            self.number_of_nodes = 2

            # Una fila por nodo: [xi]
            self.node_coordinates = numpy.array([[-1.0], [1.0]])

        else:
            self.number_of_nodes = 3

            self.node_coordinates = numpy.array([[-1.0], [1.0], [0.0]])

    def default_rule(self):
        """Regla de cuadratura de integracion completa (ver Quadrature):
        cantidad de puntos de Gauss.
        """
        if self.order == 1:
            return 2

        return 3

    def N(self, xi):
        """Funciones de forma en xi. Devuelve un vector con una componente por nodo."""
        if self.order == 1:
            return numpy.array([
                0.5 * (1.0 - xi),
                0.5 * (1.0 + xi),
            ])

        return numpy.array([
            0.5 * xi * (xi - 1.0),
            0.5 * xi * (xi + 1.0),
            1.0 - xi * xi,
        ])

    def dN(self, xi):
        """Derivadas naturales en xi. Devuelve una matriz de 1 x nodos:
        fila 0 = dN/dxi, una columna por nodo.
        """
        if self.order == 1:
            return numpy.array([[-0.5, 0.5]])

        return numpy.array([[xi - 0.5, xi + 0.5, -2.0 * xi]])
