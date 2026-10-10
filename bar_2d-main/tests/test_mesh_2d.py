"""Verificaciones de la malla 2D y de su lector.

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_mesh_2d
"""

import os
import tempfile

import numpy

from source.mesh.mesh_2d import Mesh2D
from source.reference_elements.t3 import T3
from source.reference_elements.q4 import Q4
from source.reference_elements.t6 import T6
from source.reference_elements.q8 import Q8


# (archivo, elemento de referencia, cantidad de nodos, cantidad de elementos)
PATCHES = [
    ("meshes/patch_t3.mesh", T3(), 16, 18),
    ("meshes/patch_q4.mesh", Q4(), 16, 9),
    ("meshes/patch_t6.mesh", T6(), 49, 18),
    ("meshes/patch_q8.mesh", Q8(), 40, 9),
]

VALID = """# n_nodes n_elements dim
4 1 2
MATERIAL 1 plane_strain 21e9 0.25 2400 1.0
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
1 Q4 1 1 1 2 3 4
"""


def read_text(text):
    # Escribe el texto en un archivo temporal, lo lee y lo borra
    handle, path = tempfile.mkstemp(suffix=".mesh")
    os.close(handle)

    try:
        with open(path, "w") as file:
            file.write(text)

        mesh = Mesh2D()
        mesh.read(path)
    finally:
        os.remove(path)

    return mesh


def test_read_patches():
    for filename, element, n_nodes, n_elements in PATCHES:
        mesh = Mesh2D()
        mesh.read(filename)

        assert mesh.element_type == element.name
        assert mesh.number_of_nodes() == n_nodes
        assert mesh.number_of_elements() == n_elements
        assert mesh.number_of_dofs() == 2 * n_nodes

        for e in range(mesh.number_of_elements()):
            assert len(mesh.element_node_ids(e)) == element.number_of_nodes
            assert mesh.element_coordinates(e).shape == (element.number_of_nodes, 2)


def test_arrays_for_element_set():
    # La malla entrega matrices listas para ElementSet(ref, coordenadas, conectividad)
    for filename, element, n_nodes, n_elements in PATCHES:
        mesh = Mesh2D()
        mesh.read(filename)

        coordinates = mesh.node_coordinates()
        connectivity = mesh.connectivity_array()

        assert coordinates.shape == (n_nodes, 2)
        assert connectivity.shape == (n_elements, element.number_of_nodes)
        assert connectivity.min() == 0
        assert connectivity.max() == n_nodes - 1

        # Indexar con la conectividad da lo mismo que element_coordinates
        for e in range(n_elements):
            assert numpy.allclose(coordinates[connectivity[e]], mesh.element_coordinates(e))


def test_patch_area():
    # Las areas de los elementos suman el area del parche (3 x 3 = 9)
    for filename, element, _, _ in PATCHES:
        mesh = Mesh2D()
        mesh.read(filename)

        total = 0.0

        for e in range(mesh.number_of_elements()):
            coordinates = mesh.element_coordinates(e)[:element.number_of_nodes]
            vertices = 3 if element.name in ("T3", "T6") else 4
            x = coordinates[:vertices, 0]
            y = coordinates[:vertices, 1]

            total = total + 0.5 * numpy.sum(x * numpy.roll(y, -1) - numpy.roll(x, -1) * y)

        assert numpy.isclose(total, 9.0)


def test_midside_nodes_are_at_midpoints():
    # En T6 y Q8 los nodos de mitad de lado estan en el punto medio del lado
    for filename, element, _, _ in PATCHES:
        if element.number_of_nodes in (3, 4):
            continue

        mesh = Mesh2D()
        mesh.read(filename)

        n_vertices = 3 if element.name == "T6" else 4

        for e in range(mesh.number_of_elements()):
            c = mesh.element_coordinates(e)

            for k in range(n_vertices):
                midpoint = 0.5 * (c[k] + c[(k + 1) % n_vertices])
                assert numpy.allclose(c[n_vertices + k], midpoint)


def test_material_and_dofs():
    mesh = read_text(VALID)

    material = mesh.material(0)

    assert material.model == "plane_strain"
    assert material.E == 21e9
    assert material.nu == 0.25
    assert material.rho == 2400.0
    assert material.thickness == 1.0

    assert mesh.node_global_dofs(2) == [4, 5]
    assert mesh.element_node_ids(0) == [0, 1, 2, 3]


if __name__ == "__main__":
    tests = [
        test_read_patches,
        test_arrays_for_element_set,
        test_patch_area,
        test_midside_nodes_are_at_midpoints,
        test_material_and_dofs,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)
