"""Resultantes de fuerzas nodales, control de contacto en la base y estatica
a mano de la presa.

Convencion de momentos: respecto de un punto O = (x_O, y_O), positivos en
sentido antihorario: M = (x - x_O) F_y - (y - y_O) F_x.
"""

from source.systems.dofs import node_dofs


def resultant(coordinates, forces, node_ids, point):
    """Suma de las fuerzas nodales de los nodos node_ids.

    forces es un vector global [Fx0, Fy0, Fx1, Fy1, ...] (sirve tanto para las
    reacciones R como para las cargas f). Devuelve (Fx, Fy, M), con M respecto
    de point = (x_O, y_O), antihorario positivo.
    """
    Fx = 0.0
    Fy = 0.0
    M = 0.0

    for node_id in node_ids:
        dof_x, dof_y = node_dofs(node_id)
        x, y = coordinates[node_id]

        Fx = Fx + forces[dof_x]
        Fy = Fy + forces[dof_y]
        M = M + (x - point[0]) * forces[dof_y] - (y - point[1]) * forces[dof_x]

    return Fx, Fy, M


def base_contact_table(coordinates, R, base_nodes, mu):
    """Control de contacto en la base. Devuelve una lista con un diccionario
    por nodo de la base, ordenada por x:

        node:    numero global de nodo
        x:       coordenada x
        Rx, Ry:  reacciones del nodo
        ratio:   |Rx| / Ry (infinito si Ry <= 0)
        tension: True si el nodo esta traccionado (Ry < 0)
        sliding: True si hay deslizamiento local (|Rx| > mu Ry)

    El empotramiento equivale a un contacto real con rozamiento solo si
    ninguna de las dos banderas se activa.
    """
    # Ordena los nodos por su coordenada x
    ordered = sorted(base_nodes, key=lambda node_id: coordinates[node_id][0])

    table = []

    for node_id in ordered:
        dof_x, dof_y = node_dofs(node_id)
        Rx = R[dof_x]
        Ry = R[dof_y]

        if Ry > 0.0:
            ratio = abs(Rx) / Ry
        else:
            ratio = float("inf")

        table.append({
            "node": int(node_id),
            "x": float(coordinates[node_id][0]),
            "Rx": float(Rx),
            "Ry": float(Ry),
            "ratio": ratio,
            "tension": bool(Ry < 0.0),
            "sliding": bool(abs(Rx) > mu * Ry),
        })

    return table


def dam_statics(B, c, H, h, rho, rho_w, g, thickness=1.0):
    """Estatica a mano de la presa: seccion trapecial con el paramento aguas
    arriba vertical en x = 0, base B, cresta c, altura H y agua hasta h.
    Talon en (0, 0) y pie en (B, 0), que es el punto de vuelco.

    Devuelve un diccionario con:

        A:          area de la seccion
        x_G:        posicion del baricentro, medida desde el talon
        W:          peso
        E:          empuje del agua (aplicado a h / 3 de la base)
        arm_W:      brazo del peso respecto del pie (B - x_G)
        M_W:        momento del peso respecto del pie (> 0, estabilizante)
        M_E:        momento del empuje respecto del pie (< 0, desestabilizante)
        FS:         factor de seguridad al vuelco, M_W / |M_E|
        mu_min:     rozamiento minimo para que no deslice, E / W
        x_r:        distancia de la resultante al pie
        e:          excentricidad de la resultante respecto del centro de la base
        sigma_toe:  tension lineal de compresion de la base en el pie
        sigma_heel: tension lineal de compresion de la base en el talon
    """
    A = (B + c) * H / 2.0
    x_G = (B * B + B * c + c * c) / (3.0 * (B + c))

    W = rho * g * A * thickness
    E = 0.5 * rho_w * g * h * h * thickness

    arm_W = B - x_G
    M_W = W * arm_W
    M_E = -E * h / 3.0

    x_r = (M_W + M_E) / W
    e = B / 2.0 - x_r

    # Distribucion lineal de tensiones en la base (toda comprimida si e <= B / 6)
    sigma_mean = W / (B * thickness)

    return {
        "A": A,
        "x_G": x_G,
        "W": W,
        "E": E,
        "arm_W": arm_W,
        "M_W": M_W,
        "M_E": M_E,
        "FS": M_W / abs(M_E),
        "mu_min": E / W,
        "x_r": x_r,
        "e": e,
        "sigma_toe": sigma_mean * (1.0 + 6.0 * e / B),
        "sigma_heel": sigma_mean * (1.0 - 6.0 * e / B),
    }
