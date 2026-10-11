import numpy


class PlaneElement:
    """Un elemento plano (T3, T6, Q4 o Q8) de un ElementSet.

    Recibe:
        element_set: conjunto al que pertenece el elemento
        element:     numero del elemento dentro del conjunto (desde 0)
        D:           matriz constitutiva de 3 x 3 (tension plana o deformacion
                     plana), con las deformaciones ordenadas [eps_x, eps_y, gamma_xy]
        thickness:   espesor

    Grados de libertad locales ordenados por nodo: [u0, v0, u1, v1, ...].

    Tiene los dos metodos que el ensamblador le pide a cualquier elemento:
    stiffness_matrix() y global_dofs().
    """

    dofs_per_node = 2

    def __init__(self, element_set, element, D, thickness):
        self.element_set = element_set
        self.element = element
        self.D = numpy.array(D, dtype=float)
        self.thickness = thickness

        self.node_global_ids = element_set.connectivity[element]

        number_of_dofs = self.dofs_per_node * element_set.nodes_per_element

        # Ke = sum_g  B^T D B * (w detJ) * t
        self.Ke = numpy.zeros((number_of_dofs, number_of_dofs))

        for g in range(element_set.number_of_points):
            B = self.B_matrix(g)

            self.Ke = self.Ke + B.transpose() @ self.D @ B * (
                element_set.wdetJ(element, g) * thickness
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
        """DOFs globales del elemento: el nodo global n tiene u en 2n y v en 2n + 1."""
        dofs = []

        for node_global_id in self.node_global_ids:
            for i in range(self.dofs_per_node):
                dofs.append(int(node_global_id) * self.dofs_per_node + i)

        return dofs

    def stiffness_matrix(self):
        return self.Ke
