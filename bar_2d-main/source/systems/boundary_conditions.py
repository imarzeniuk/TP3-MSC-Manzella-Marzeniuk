import numpy

from source.systems.dofs import node_dofs


def apply_dirichlet_bc(K, f, dof, value):
    for i in range(K.number_of_dofs):
        f[i] = f[i] - K.values[i][dof] * value

    for i in range(K.number_of_dofs):
        K.values[dof][i] = 0.0
        K.values[i][dof] = 0.0

    K.values[dof][dof] = 1.0
    f[dof] = value


def nodes_on_line(coordinates, axis, value, tol=None):
    """Indices de los nodos con coordinates[n, axis] == value (dentro de tol).

    axis = 0 para x, 1 para y. Si tol es None se usa una tolerancia relativa:
    1e-9 por la dimension mayor del modelo. Lanza ValueError si no encuentra
    ninguno.

    Como selecciona por coordenada, incluye tambien los nodos de mitad de lado
    de los elementos cuadraticos.
    """
    coordinates = numpy.asarray(coordinates, dtype=float)

    if tol is None:
        # Dimension mayor del modelo: el mayor de (x_max - x_min) e (y_max - y_min)
        size = numpy.max(coordinates.max(axis=0) - coordinates.min(axis=0))

        if size == 0.0:
            size = 1.0

        tol = 1.0e-9 * size

    node_ids = []

    for n in range(coordinates.shape[0]):
        if abs(coordinates[n, axis] - value) <= tol:
            node_ids.append(n)

    if len(node_ids) == 0:
        raise ValueError(
            "No nodes found with coordinate " + str(axis) + " = " + str(value)
        )

    return node_ids


def fixed_dofs(node_ids, components=(0, 1)):
    """DOFs globales a prescribir en los nodos node_ids.

    components = (0, 1) fija u y v; (0,) solo u; (1,) solo v.
    Devuelve una lista ordenada y sin repetidos.
    """
    dofs = []

    for node_id in node_ids:
        dofs_of_node = node_dofs(node_id)

        for component in components:
            dofs.append(dofs_of_node[component])

    # set(...) descarta los repetidos y sorted(...) devuelve la lista ordenada
    return sorted(set(dofs))


def solve_with_dirichlet(K, f, prescribed_dofs, prescribed_values):
    """Resuelve K u = f con u[prescribed_dofs] = prescribed_values por el
    metodo de particion.

    K es un arreglo de N x N de NumPy (DenseMatrix.values). Ni K ni f se
    modifican. Devuelve (u, R), con R = K u - f: las reacciones, que valen
    aproximadamente 0 en los DOFs libres.

    Con c = prescritos y l = libres:

        K_ll u_l = f_l - K_lc u_c
    """
    K = numpy.asarray(K, dtype=float)
    f = numpy.asarray(f, dtype=float)

    prescribed = numpy.array(prescribed_dofs, dtype=int)
    values = numpy.array(prescribed_values, dtype=float)

    if len(prescribed) != len(values):
        raise ValueError("Expected one prescribed value per prescribed DOF")

    if len(numpy.unique(prescribed)) != len(prescribed):
        raise ValueError("Repeated prescribed DOFs")

    number_of_dofs = len(f)

    # DOFs libres: todos menos los prescritos
    free = numpy.setdiff1d(numpy.arange(number_of_dofs), prescribed)

    # numpy.ix_ arma la submatriz con esas filas y columnas (K(l, l) en MATLAB)
    K_ll = K[numpy.ix_(free, free)]
    K_lc = K[numpy.ix_(free, prescribed)]

    u = numpy.zeros(number_of_dofs)
    u[prescribed] = values

    message = (
        "Singular system: the boundary conditions do not prevent the rigid "
        "body motions (at least 3 well placed DOFs are needed)"
    )

    try:
        u[free] = numpy.linalg.solve(K_ll, f[free] - K_lc @ values)
        condition_number = numpy.linalg.cond(K_ll, 1)
    except numpy.linalg.LinAlgError:
        raise ValueError(message)

    # Por el redondeo, una matriz singular casi nunca da error al resolver:
    # se la detecta porque su numero de condicion es enorme
    if not condition_number < 1.0e13:
        raise ValueError(message)

    # Reacciones con K y f originales
    R = K @ u - f

    return u, R
