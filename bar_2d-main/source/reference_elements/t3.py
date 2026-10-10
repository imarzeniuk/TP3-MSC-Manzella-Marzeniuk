import numpy


class T3:
    """Triangulo lineal de 3 nodos (CST) en coordenadas naturales (xi, eta).

    Numeracion local (antihoraria):

        eta
         ^
         2
         | \\
         |   \\
         0 --- 1 --> xi
    """

    name = "T3"
    number_of_nodes = 3

    # Una fila por nodo: [xi, eta]
    node_coordinates = numpy.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    def N(self, xi, eta):
        """Funciones de forma en (xi, eta). Devuelve un vector de 3 componentes."""
        return numpy.array([
            1.0 - xi - eta,
            xi,
            eta,
        ])

    def dN(self, xi, eta):
        """Derivadas naturales en (xi, eta). Devuelve una matriz de 2 x 3:
        fila 0 = dN/dxi, fila 1 = dN/deta, una columna por nodo.

        En el T3 son constantes (no dependen del punto), pero se mantienen los
        argumentos para que todos los elementos tengan la misma interfaz.
        """
        return numpy.array([
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ])
