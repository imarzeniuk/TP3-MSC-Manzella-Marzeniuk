"""Verificaciones de la salida a ParaView.

Los tipos de celda y los controles de tamano no necesitan meshio. La escritura
del .vtu se prueba solo si meshio esta instalado (si no, se omite).

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_paraview_2d
"""

import os
import tempfile

import numpy

from source.mesh.mesh_2d import Mesh2D
from source.output.paraview_2d import cell_type, check_sizes, write_results_2d


MESHES = [
    "meshes/patch_t3.mesh",
    "meshes/patch_q4.mesh",
    "meshes/patch_t6.mesh",
    "meshes/patch_q8.mesh",
]


def read(filename):
    mesh = Mesh2D()
    mesh.read(filename)

    return mesh


def raises_value_error(function):
    try:
        function()
    except ValueError:
        return True

    return False


def test_cell_types_with_aliases():
    assert cell_type("CST") == "triangle"
    assert cell_type("T3") == "triangle"
    assert cell_type("LST") == "triangle6"
    assert cell_type("T6") == "triangle6"
    assert cell_type("Q4") == "quad"
    assert cell_type("Q8") == "quad8"


def test_sizes_are_checked():
    mesh = read("meshes/patch_q4.mesh")  # 16 nodos, 9 elementos
    u = numpy.zeros(32)

    assert not raises_value_error(
        lambda: check_sizes(mesh, u, {"a": numpy.zeros(16)}, {"b": numpy.zeros(9)})
    )
    # u con otra cantidad de valores
    assert raises_value_error(lambda: check_sizes(mesh, numpy.zeros(30), None, None))
    # point_data con un valor por elemento en vez de por nodo
    assert raises_value_error(lambda: check_sizes(mesh, u, {"a": numpy.zeros(9)}, None))
    # cell_data con un valor por nodo en vez de por elemento
    assert raises_value_error(lambda: check_sizes(mesh, u, None, {"b": numpy.zeros(16)}))


def test_write_and_read_back():
    try:
        import meshio
    except ImportError:
        print("    (meshio no esta instalado: se omite la escritura del .vtu)")
        return

    scale = 10.0

    for filename in MESHES:
        mesh = read(filename)
        coordinates = mesh.node_coordinates()

        # Desplazamiento cualquiera, pasado como lista para probar la conversion
        u = numpy.zeros(mesh.number_of_dofs())
        u[0::2] = 1.0e-3 * coordinates[:, 1]
        u[1::2] = -2.0e-3 * coordinates[:, 0]

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "results", "patch.vtu")

            write_results_2d(
                path, mesh, list(u), scale=scale,
                point_data={"node_value": numpy.arange(mesh.number_of_nodes(), dtype=float)},
                cell_data={"cell_value": numpy.ones(mesh.number_of_elements())},
            )

            result = meshio.read(path)

        displacement = u.reshape(-1, 2)

        assert result.cells[0].type == cell_type(mesh.element_type)
        assert numpy.array_equal(result.cells[0].data, mesh.connectivity_array())
        assert numpy.allclose(result.points[:, :2], coordinates + scale * displacement)
        assert numpy.allclose(result.point_data["displacement"][:, :2], displacement)
        assert "node_value" in result.point_data
        assert "cell_value" in result.cell_data


if __name__ == "__main__":
    tests = [
        test_cell_types_with_aliases,
        test_sizes_are_checked,
        test_write_and_read_back,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)
