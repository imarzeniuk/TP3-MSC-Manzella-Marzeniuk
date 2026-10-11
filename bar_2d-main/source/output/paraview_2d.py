import os

import meshio


# Tipo de celda de meshio para cada tipo de elemento. El orden de los nodos
# (vertices y despues mitades de lado) coincide con el de nuestros elementos.
CELL_TYPES = {
    "T3": "triangle",
    "T6": "triangle6",
    "Q4": "quad",
    "Q8": "quad8",
}


def write_results_2d(filename, mesh, u, scale=1.0, point_data=None, cell_data=None):
    """Escribe un .vtu con la malla en la configuracion deformada.

    u          : vector de desplazamientos [u0, v0, u1, v1, ...]
    scale      : factor de escala; los nodos se dibujan en x + scale * u
    point_data : diccionario nombre -> arreglo con un valor por nodo (opcional)
    cell_data  : diccionario nombre -> arreglo con un valor por elemento (opcional)

    El desplazamiento real (sin escalar) se guarda como "displacement".
    """
    directory = os.path.dirname(filename)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    # u.reshape(-1, 2): una fila [u, v] por nodo ("-1" deja que NumPy calcule las filas)
    displacement = u.reshape(-1, 2)
    deformed = mesh.node_coordinates() + scale * displacement

    # meshio trabaja en 3D: se agrega la columna z = 0
    points = [[p[0], p[1], 0.0] for p in deformed]
    displacement_3d = [[d[0], d[1], 0.0] for d in displacement]

    cells = [(CELL_TYPES[mesh.element_type], mesh.connectivity_array())]

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
