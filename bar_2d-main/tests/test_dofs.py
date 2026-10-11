"""Verificaciones de la numeracion de grados de libertad.

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_dofs
"""

import numpy

from source.systems.dofs import DOFS_PER_NODE, node_dofs, element_dofs
from source.mesh.mesh_2d import Mesh2D
from source.materials.plane_elastic import PlaneElastic
from source.elements.element_set import ElementSet
from source.elements.plane_element import PlaneElement


def test_node_dofs():
    assert DOFS_PER_NODE == 2
    assert node_dofs(0) == [0, 1]
    assert node_dofs(7) == [14, 15]
    # Un id de nodo que sale de un arreglo de NumPy da enteros de Python
    assert node_dofs(numpy.int64(3)) == [6, 7]


def test_element_dofs():
    assert element_dofs([2, 0, 5]) == [4, 5, 0, 1, 10, 11]


def test_mesh_and_elements_agree():
    # La malla y los elementos usan la misma numeracion
    for filename in ("meshes/patch_t3.mesh", "meshes/patch_q8.mesh"):
        mesh = Mesh2D()
        mesh.read(filename)

        material = PlaneElastic.from_spec(mesh.material(0))
        element_set = ElementSet(
            mesh.reference_element(), mesh.node_coordinates(), mesh.connectivity_array()
        )

        for e in range(mesh.number_of_elements()):
            expected = []

            for node_global_id in mesh.element_node_ids(e):
                expected.extend(mesh.node_global_dofs(node_global_id))

            assert PlaneElement(element_set, e, material).global_dofs() == expected


if __name__ == "__main__":
    tests = [
        test_node_dofs,
        test_element_dofs,
        test_mesh_and_elements_agree,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)
