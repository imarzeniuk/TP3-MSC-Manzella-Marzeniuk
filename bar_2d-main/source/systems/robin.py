import numpy

from source.quadrature.quadrature import Quadrature
from source.reference_elements.line import Line
from source.systems.dofs import DOFS_PER_NODE, element_dofs


def robin_base_matrix(element_set, materials, k_normal, k_tangent, base_y=0.0, tol=1.0e-9):
    """Matriz global de los resortes de la base (condicion de Robin):
    K_robin = integral de t * N^T diag(k_t, k_n) N ds sobre los lados con y = base_y.

    k_normal (vertical) y k_tangent (horizontal) son rigideces por unidad de
    area, en N/m3. Se suma a K; la fuerza de los resortes sobre la presa es
    -K_robin u.
    """
    number_of_dofs = DOFS_PER_NODE * element_set.node_coordinates.shape[0]
    K_robin = numpy.zeros((number_of_dofs, number_of_dofs))

    rule = Quadrature("line", 3)

    for e in range(element_set.number_of_elements):
        thickness = materials[e].thickness

        for edge in element_set.reference_element.edges:
            node_ids = element_set.connectivity[e][list(edge)]
            coordinates = element_set.node_coordinates[node_ids]

            if numpy.any(numpy.abs(coordinates[:, 1] - base_y) > tol):
                continue

            line = Line("L" + str(len(edge)))
            Ke = numpy.zeros((DOFS_PER_NODE * len(edge), DOFS_PER_NODE * len(edge)))

            for s in range(rule.number_of_points):
                xi = rule.points[s][0]

                N = line.N(xi)
                tangent = line.dN(xi)[0] @ coordinates
                weight = rule.weights[s] * numpy.linalg.norm(tangent) * thickness

                for a in range(len(edge)):
                    for b in range(len(edge)):
                        Ke[2 * a, 2 * b] = Ke[2 * a, 2 * b] + k_tangent * N[a] * N[b] * weight
                        Ke[2 * a + 1, 2 * b + 1] = Ke[2 * a + 1, 2 * b + 1] + k_normal * N[a] * N[b] * weight

            dofs = element_dofs(node_ids)

            for i in range(len(dofs)):
                for j in range(len(dofs)):
                    K_robin[dofs[i], dofs[j]] = K_robin[dofs[i], dofs[j]] + Ke[i, j]

    return K_robin
