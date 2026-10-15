import numpy

from source.quadrature.quadrature import Quadrature
from source.reference_elements.line import Line
from source.systems.dofs import DOFS_PER_NODE, element_dofs


def body_force_vector(element_set, materials, g):
    """Vector global de fuerzas por el peso propio: fe = integral N^T b t dA,
    con b = [0, -rho g]. materials[e] es el material del elemento e.
    """
    f = numpy.zeros(DOFS_PER_NODE * element_set.node_coordinates.shape[0])

    for e in range(element_set.number_of_elements):
        material = materials[e]
        fe = numpy.zeros(DOFS_PER_NODE * element_set.nodes_per_element)

        for gp in range(element_set.number_of_points):
            weight = element_set.wdetJ(e, gp) * material.thickness

            for a in range(element_set.nodes_per_element):
                fe[2 * a + 1] = fe[2 * a + 1] - material.rho * g * element_set.shape(gp, a) * weight

        dofs = element_dofs(element_set.connectivity[e])

        for i in range(len(dofs)):
            f[dofs[i]] = f[dofs[i]] + fe[i]

    return f


def face_pressure_vector(element_set, materials, pressure, water_level,
                         wall_x=0.0, tol=1.0e-9):
    """Vector global de fuerzas por una presion pressure(y) sobre el paramento
    vertical x = wall_x: fe = integral N^T p(y) n t ds, aplicada solo por debajo
    de water_level. Los lados cortados por el nivel del agua se integran solo
    en la parte mojada.

    pressure es una funcion de y que devuelve la presion en Pa (positiva
    empujando hacia adentro de la presa).
    """
    f = numpy.zeros(DOFS_PER_NODE * element_set.node_coordinates.shape[0])

    rule = Quadrature("line", 3)

    for e in range(element_set.number_of_elements):
        thickness = materials[e].thickness

        for edge in element_set.reference_element.edges:
            node_ids = element_set.connectivity[e][list(edge)]
            coordinates = element_set.node_coordinates[node_ids]

            if numpy.any(numpy.abs(coordinates[:, 0] - wall_x) > tol):
                continue

            y0 = coordinates[0, 1]
            y1 = coordinates[1, 1]

            if y0 >= water_level and y1 >= water_level:
                continue

            xi_a = -1.0
            xi_b = 1.0

            if y0 > water_level:
                xi_a = -1.0 + 2.0 * (water_level - y0) / (y1 - y0)
            elif y1 > water_level:
                xi_b = -1.0 + 2.0 * (water_level - y0) / (y1 - y0)

            line = Line("L" + str(len(edge)))
            fe = numpy.zeros(DOFS_PER_NODE * len(edge))

            for s in range(rule.number_of_points):
                xi = 0.5 * (xi_a + xi_b) + 0.5 * (xi_b - xi_a) * rule.points[s][0]
                weight = 0.5 * (xi_b - xi_a) * rule.weights[s]

                N = line.N(xi)
                tangent = line.dN(xi)[0] @ coordinates
                p = pressure(N @ coordinates[:, 1])

                # La presion empuja hacia adentro: p * (-ty, tx) * dxi
                for a in range(len(edge)):
                    fe[2 * a] = fe[2 * a] - N[a] * p * tangent[1] * weight * thickness
                    fe[2 * a + 1] = fe[2 * a + 1] + N[a] * p * tangent[0] * weight * thickness

            dofs = element_dofs(node_ids)

            for i in range(len(dofs)):
                f[dofs[i]] = f[dofs[i]] + fe[i]

    return f


def hydrostatic_vector(element_set, materials, water_density, g, water_level,
                       wall_x=0.0, tol=1.0e-9):
    """Vector global de fuerzas por la presion hidrostatica del agua sobre el
    paramento vertical x = wall_x: p(y) = rho_w g (h - y) por debajo del nivel
    h y 0 por encima (carga triangular).
    """
    def pressure(y):
        return water_density * g * (water_level - y)

    return face_pressure_vector(element_set, materials, pressure, water_level, wall_x, tol)


def uniform_pressure_vector(element_set, materials, pressure_value, water_level,
                            wall_x=0.0, tol=1.0e-9):
    """Vector global de fuerzas por una presion uniforme pressure_value sobre el
    paramento vertical x = wall_x, desde la base hasta water_level (carga
    rectangular).
    """
    def pressure(y):
        return pressure_value

    return face_pressure_vector(element_set, materials, pressure, water_level, wall_x, tol)
