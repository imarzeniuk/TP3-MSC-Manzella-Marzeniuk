import numpy

from source.systems.dofs import DOFS_PER_NODE, element_dofs


class PlaneElement:
    """Un elemento plano (CST, LST, Q4 o Q8) de un ElementSet.

    Recibe:
        element_set: conjunto al que pertenece el elemento
        element:     numero del elemento dentro del conjunto (desde 0)
        material:    material plano (PlaneElastic). De el se toman la matriz
                     constitutiva D de 3 x 3 (tension plana o deformacion plana,
                     con las deformaciones ordenadas [eps_x, eps_y, gamma_xy]) y
                     el espesor, para que ambos salgan de un solo lugar.

    Grados de libertad locales ordenados por nodo: [u0, v0, u1, v1, ...].

    Tiene los dos metodos que el ensamblador le pide a cualquier elemento:
    stiffness_matrix() y global_dofs().
    """

    dofs_per_node = DOFS_PER_NODE

    def __init__(self, element_set, element, material):
        self.element_set = element_set
        self.element = element
        self.material = material
        self.D = numpy.array(material.D(), dtype=float)
        self.thickness = material.thickness

        self.node_global_ids = element_set.connectivity[element]

        number_of_dofs = self.dofs_per_node * element_set.nodes_per_element

        # Ke = sum_g  B^T D B * (w detJ) * t
        self.Ke = numpy.zeros((number_of_dofs, number_of_dofs))

        for g in range(element_set.number_of_points):
            B = self.B_matrix(g)

            self.Ke = self.Ke + B.transpose() @ self.D @ B * (
                element_set.wdetJ(element, g) * self.thickness
            )

    def B_matrix(self, gauss_point):
        """Matriz deformacion-desplazamiento de 3 x (2 * nodos) en el punto de
        Gauss. El bloque de cada nodo es

            [[dN/dx, 0    ],
             [0,     dN/dy],
             [dN/dy, dN/dx]]
        """
        element_set = self.element_set

        B = numpy.zeros((3, self.dofs_per_node * element_set.nodes_per_element))

        for a in range(element_set.nodes_per_element):
            dN_dx = element_set.dshape(self.element, gauss_point, a, 0)
            dN_dy = element_set.dshape(self.element, gauss_point, a, 1)

            B[0, 2 * a] = dN_dx
            B[1, 2 * a + 1] = dN_dy
            B[2, 2 * a] = dN_dy
            B[2, 2 * a + 1] = dN_dx

        return B

    def global_dofs(self):
        """DOFs globales del elemento: el nodo global n tiene u en 2n y v en 2n + 1
        (ver source/systems/dofs.py, que es la unica definicion de la numeracion).
        """
        return element_dofs(self.node_global_ids)

    def stiffness_matrix(self):
        return self.Ke

    def displacements(self, u):
        """Desplazamientos del elemento, u_e = u[global_dofs()], a partir del
        vector global u.
        """
        return numpy.asarray(u, dtype=float)[self.global_dofs()]

    def strain(self, u, gauss_point):
        """Deformacion eps = B u_e = [eps_x, eps_y, gamma_xy] en el punto de Gauss."""
        return self.B_matrix(gauss_point) @ self.displacements(u)

    def stress(self, u, gauss_point):
        """Tension sigma = D B u_e = [sigma_x, sigma_y, tau_xy] en el punto de Gauss."""
        return self.D @ self.strain(u, gauss_point)
