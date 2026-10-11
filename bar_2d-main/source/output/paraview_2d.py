import os

import numpy

from source.reference_elements import make_reference_element


# Tipo de celda de meshio para cada elemento (por su nombre canonico). El orden
# de los nodos (vertices y despues mitades de lado) coincide con el de nuestros
# elementos. Los alias T3 y T6 se resuelven con make_reference_element.
CELL_TYPES = {
    "CST": "triangle",
    "LST": "triangle6",
    "Q4": "quad",
    "Q8": "quad8",
}


def cell_type(element_type):
    """Tipo de celda de meshio para un nombre de elemento (acepta alias)."""
    return CELL_TYPES[make_reference_element(element_type).name]


def check_sizes(mesh, u, point_data, cell_data):
    """Lanza ValueError si los datos no tienen el tamano de la malla."""
    n_nodes = mesh.number_of_nodes()
    n_elements = mesh.number_of_elements()

    if u.shape != (mesh.number_of_dofs(),):
        raise ValueError(
            "Expected u with " + str(mesh.number_of_dofs()) + " values, got shape "
            + str(u.shape)
        )

    if point_data is not None:
        for name in point_data:
            if len(point_data[name]) != n_nodes:
                raise ValueError("point_data '" + name + "' needs one value per node")

    if cell_data is not None:
        for name in cell_data:
            if len(cell_data[name]) != n_elements:
                raise ValueError("cell_data '" + name + "' needs one value per element")


def write_results_2d(filename, mesh, u, scale=1.0, point_data=None, cell_data=None):
    """Escribe un .vtu con la malla en la configuracion deformada.

    u          : vector de desplazamientos [u0, v0, u1, v1, ...]
    scale      : factor de escala; los nodos se dibujan en x + scale * u
    point_data : diccionario nombre -> arreglo con un valor por nodo (opcional)
    cell_data  : diccionario nombre -> arreglo con un valor por elemento (opcional)

    El desplazamiento real (sin escalar) se guarda como "displacement".
    """
    # meshio se importa aca y no al principio del modulo: asi cell_type y
    # check_sizes (y sus tests) funcionan aunque meshio no este instalado
    import meshio

    # u puede llegar como lista o como arreglo: se convierte a arreglo de NumPy
    u = numpy.asarray(u, dtype=float)

    check_sizes(mesh, u, point_data, cell_data)

    directory = os.path.dirname(filename)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    # u.reshape(-1, 2): una fila [u, v] por nodo ("-1" deja que NumPy calcule las filas)
    displacement = u.reshape(-1, 2)
    deformed = mesh.node_coordinates() + scale * displacement

    # meshio trabaja en 3D: se agrega la columna z = 0
    points = [[p[0], p[1], 0.0] for p in deformed]
    displacement_3d = [[d[0], d[1], 0.0] for d in displacement]

    cells = [(cell_type(mesh.element_type), mesh.connectivity_array())]

    all_point_data = {"displacement": displacement_3d}

    if point_data is not None:
        all_point_data.update(point_data)

    all_cell_data = {}

    if cell_data is not None:
        for name in cell_data:
            all_cell_data[name] = [cell_data[name]]

    output_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=all_point_data,
        cell_data=all_cell_data,
    )

    output_mesh.write(filename)
