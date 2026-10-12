"""Calculo de tensiones a partir de los desplazamientos.

En todo el modulo cada tension tiene 4 componentes, en el orden de
STRESS_NAMES: [sigma_x, sigma_y, tau_xy, sigma_z]. Traccion positiva.

Las tensiones se calculan en los puntos de Gauss (donde son mas precisas) y
de ahi se obtienen los promedios por elemento y los valores en los nodos.
"""

import numpy

from source.reference_elements import make_reference_element


STRESS_NAMES = ["sigma_x", "sigma_y", "tau_xy", "sigma_z"]


def gauss_point_stresses(elements, u):
    """Tensiones en los puntos de Gauss: arreglo de
    (elementos, puntos de Gauss, 4) con [sigma_x, sigma_y, tau_xy, sigma_z].

    elements es la lista de PlaneElement (todos del mismo ElementSet) y u el
    vector global de desplazamientos. sigma_z se calcula con el material de
    cada elemento.
    """
    number_of_points = elements[0].element_set.number_of_points

    sigma = numpy.zeros((len(elements), number_of_points, 4))

    for e in range(len(elements)):
        element = elements[e]

        for g in range(number_of_points):
            in_plane = element.stress(u, g)

            # [e, g, 0:3]: las tres primeras componentes (el 3 no se incluye)
            sigma[e, g, 0:3] = in_plane
            sigma[e, g, 3] = element.material.out_of_plane_stress(in_plane)

    return sigma


def gauss_point_coordinates(element_set):
    """Coordenadas [x, y] de los puntos de Gauss: arreglo de
    (elementos, puntos de Gauss, 2), con x_g = sum_a N_a(g) x_a.
    """
    coordinates = numpy.zeros(
        (element_set.number_of_elements, element_set.number_of_points, 2)
    )

    for e in range(element_set.number_of_elements):
        # Coordenadas de los nodos del elemento: (nodos por elemento) x 2
        element_coordinates = element_set.node_coordinates[element_set.connectivity[e]]

        # element_set.N es (puntos de Gauss) x (nodos por elemento)
        coordinates[e] = element_set.N @ element_coordinates

    return coordinates


def principal_stresses(sigma):
    """Tensiones principales a partir de sigma, cuyo ultimo eje es
    [sigma_x, sigma_y, tau_xy, sigma_z]. Devuelve un diccionario de arreglos
    con la forma de sigma sin el ultimo eje:

        sigma_1, sigma_2: principales en el plano (sigma_1 >= sigma_2)
        theta_1:          angulo de sigma_1 respecto del eje x, en radianes
        sigma_max:        maxima tension principal, max(sigma_1, sigma_z)
        sigma_min:        minima tension principal, min(sigma_2, sigma_z)
    """
    sigma = numpy.asarray(sigma, dtype=float)

    sigma_x = sigma[..., 0]
    sigma_y = sigma[..., 1]
    tau_xy = sigma[..., 2]
    sigma_z = sigma[..., 3]

    # Centro y radio del circulo de Mohr
    center = 0.5 * (sigma_x + sigma_y)
    radius = numpy.sqrt((0.5 * (sigma_x - sigma_y)) ** 2 + tau_xy ** 2)

    sigma_1 = center + radius
    sigma_2 = center - radius

    return {
        "sigma_1": sigma_1,
        "sigma_2": sigma_2,
        "theta_1": 0.5 * numpy.arctan2(2.0 * tau_xy, sigma_x - sigma_y),
        "sigma_max": numpy.maximum(sigma_1, sigma_z),
        "sigma_min": numpy.minimum(sigma_2, sigma_z),
    }


def element_average(element_set, sigma_gp):
    """Promedio de las tensiones en cada elemento, ponderado por area:
    sum_g sigma_g (w detJ)_g / sum_g (w detJ)_g. Devuelve (elementos, 4).
    """
    average = numpy.zeros((element_set.number_of_elements, sigma_gp.shape[2]))

    for e in range(element_set.number_of_elements):
        area = 0.0

        for g in range(element_set.number_of_points):
            average[e] = average[e] + sigma_gp[e, g] * element_set.wdetJ(e, g)
            area = area + element_set.wdetJ(e, g)

        average[e] = average[e] / area

    return average


def fit_basis(reference_element, number_of_points):
    """Funciones con las que se ajustan los valores de los puntos de Gauss.
    Devuelve una funcion (xi, eta) -> vector con el valor de cada funcion de
    la base en ese punto.

    Se usa la base del propio elemento si hay al menos tantos puntos de Gauss
    como funciones; si no, la base lineal de la misma familia (CST o Q4); y si
    tampoco alcanza (1 punto), una constante.
    """
    if number_of_points >= reference_element.number_of_nodes:
        return reference_element.N

    if reference_element.family == "triangle":
        linear_element = make_reference_element("CST")
    else:
        linear_element = make_reference_element("Q4")

    if number_of_points >= linear_element.number_of_nodes:
        return linear_element.N

    return lambda xi, eta: numpy.array([1.0])


def extrapolation_matrix(reference_element, quadrature):
    """Matriz E de (nodos del elemento) x (puntos de Gauss) tal que
    sigma_nodos = E @ sigma_gauss.

    Se ajustan por minimos cuadrados los valores de los puntos de Gauss con
    una base de funciones de forma (ver fit_basis) y se evalua ese ajuste en
    los nodos:

        A[g, j]  = N_j(xi_g, eta_g)     base en los puntos de Gauss
        Nn[a, j] = N_j(xi_a, eta_a)     base en los nodos del elemento
        E = Nn @ pinv(A)

    Cada fila de E suma 1: un campo constante se extrapola sin cambios.
    """
    basis = fit_basis(reference_element, quadrature.number_of_points)

    A = []

    for xi, eta in quadrature.points:
        A.append(basis(xi, eta))

    Nn = []

    for xi, eta in reference_element.node_coordinates:
        Nn.append(basis(xi, eta))

    # pinv es la pseudoinversa: da el ajuste por minimos cuadrados (y la
    # inversa comun cuando A es cuadrada)
    return numpy.array(Nn) @ numpy.linalg.pinv(numpy.array(A))


def nodal_stresses(element_set, sigma_gp):
    """Tensiones en los nodos de la malla: arreglo de (nodos de la malla, 4).

    Extrapola sigma_gp a los nodos de cada elemento con extrapolation_matrix y
    en cada nodo promedia (promedio aritmetico) los valores que llegan de los
    elementos que lo comparten.

    Se extrapolan las componentes. Las tensiones principales en los nodos se
    calculan despues, con principal_stresses sobre este resultado: no son
    lineales en las componentes, asi que no se deben extrapolar directamente.
    """
    E = extrapolation_matrix(element_set.reference_element, element_set.quadrature)

    number_of_nodes = element_set.node_coordinates.shape[0]

    total = numpy.zeros((number_of_nodes, sigma_gp.shape[2]))
    count = numpy.zeros(number_of_nodes)

    for e in range(element_set.number_of_elements):
        # (nodos por elemento) x 4: valores extrapolados a los nodos del elemento
        element_values = E @ sigma_gp[e]

        for a in range(element_set.nodes_per_element):
            node_id = element_set.connectivity[e][a]

            total[node_id] = total[node_id] + element_values[a]
            count[node_id] = count[node_id] + 1.0

    # Un nodo que no pertenece a ningun elemento queda con tension 0
    count[count == 0.0] = 1.0

    # count[:, None] convierte el vector en columna para dividir fila por fila
    return total / count[:, None]
