import numpy

from source.quadrature.quadrature import Quadrature


class ElementSet:
    """Conjunto de elementos del mismo tipo (todos T3, todos Q4, etc.).

    Recibe:
        reference_element: objeto T3, T6, Q4 o Q8
        node_coordinates:  matriz de (nodos de la malla) x 2 con [x, y] de cada nodo
        connectivity:      matriz de (elementos) x (nodos por elemento) con los
                           numeros globales de nodo de cada elemento (desde 0)

    Todos los indices empiezan en 0:
        element: numero de elemento dentro del conjunto
        gauss_point: numero de punto de Gauss dentro del elemento
        node: numero LOCAL de nodo dentro del elemento
        variable: 0 para x, 1 para y

    Las funciones de forma, sus derivadas fisicas y w * det(J) se calculan una
    sola vez al construir el objeto; shape, dshape y wdetJ solo las consultan.
    """

    def __init__(self, reference_element, node_coordinates, connectivity):
        self.reference_element = reference_element
        self.quadrature = Quadrature(reference_element)
        self.node_coordinates = numpy.array(node_coordinates, dtype=float)
        self.connectivity = numpy.array(connectivity, dtype=int)

        self.number_of_elements = self.connectivity.shape[0]
        self.nodes_per_element = reference_element.number_of_nodes
        self.number_of_points = self.quadrature.number_of_points

        if self.connectivity.shape[1] != self.nodes_per_element:
            raise ValueError(
                "Expected " + str(self.nodes_per_element) + " nodes per element"
            )

        n_elements = self.number_of_elements
        n_nodes = self.nodes_per_element
        n_points = self.number_of_points

        # N y dN naturales solo dependen del punto de Gauss, no del elemento
        self.N = numpy.zeros((n_points, n_nodes))
        dN_natural = numpy.zeros((n_points, 2, n_nodes))

        for g in range(n_points):
            xi, eta = self.quadrature.points[g]

            self.N[g] = reference_element.N(xi, eta)
            dN_natural[g] = reference_element.dN(xi, eta)

        # Las derivadas fisicas y det(J) dependen de la geometria de cada elemento
        self.dN = numpy.zeros((n_elements, n_points, 2, n_nodes))
        self.wdetJ_values = numpy.zeros((n_elements, n_points))

        for e in range(n_elements):
            # Coordenadas [x, y] de los nodos del elemento: (nodos por elemento) x 2
            coordinates = self.node_coordinates[self.connectivity[e]]

            for g in range(n_points):
                # J = [[dx/dxi,  dy/dxi ],
                #      [dx/deta, dy/deta]]
                J = dN_natural[g] @ coordinates
                detJ = numpy.linalg.det(J)

                if detJ <= 0.0:
                    raise ValueError(
                        "Expected positive det(J) in element " + str(e)
                        + " (check node ordering: must be counterclockwise)"
                    )

                # dN/dx = J^-1 * dN/dxi (se resuelve el sistema en vez de invertir J)
                self.dN[e, g] = numpy.linalg.solve(J, dN_natural[g])
                self.wdetJ_values[e, g] = self.quadrature.weights[g] * detJ

    def shape(self, gauss_point, node):
        """Funcion de forma del nodo evaluada en el punto de Gauss."""
        return self.N[gauss_point, node]

    def dshape(self, element, gauss_point, node, variable):
        """Derivada fisica de la funcion de forma del nodo en el punto de Gauss
        del elemento, respecto de x (variable = 0) o de y (variable = 1).
        """
        return self.dN[element, gauss_point, variable, node]

    def wdetJ(self, element, gauss_point):
        """Peso del punto de Gauss por det(J) del elemento en ese punto."""
        return self.wdetJ_values[element, gauss_point]
