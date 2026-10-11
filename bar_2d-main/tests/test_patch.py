"""Patch test de punta a punta para los cuatro tipos de elemento.

Se impone en el borde del parche [0, 3] x [0, 3] un campo de desplazamientos
lineal (deformacion constante). La solucion debe reproducir ese campo en los
nodos interiores y la tension constante exacta en todos los puntos de Gauss.

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_patch
"""

import numpy

from source.mesh.mesh_2d import Mesh2D
from source.reference_elements import make_reference_element
from source.materials.plane_elastic import PlaneElastic
from source.elements.element_set import ElementSet
from source.elements.plane_element import PlaneElement
from source.systems.dense_matrix import DenseMatrix
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.solvers.linear import solve


MESHES = [
    "meshes/patch_t3.mesh",
    "meshes/patch_q4.mesh",
    "meshes/patch_t6.mesh",
    "meshes/patch_q8.mesh",
]

MODELS = ["plane_stress", "plane_strain"]

# Campo impuesto: u = A x + C y, v = B y  ->  eps = [A, B, C] constante
A = 1.0e-3
B = -2.0e-3
C = 0.5e-3


def exact_displacement(x, y):
    return A * x + C * y, B * y


def run_patch(filename, model):
    """Resuelve el parche y devuelve (error en desplazamientos, error en
    tensiones), ambos relativos.
    """
    mesh = Mesh2D()
    mesh.read(filename)

    spec = mesh.material(0)
    material = PlaneElastic(spec.E, spec.nu, model, spec.rho, spec.thickness)
    D = material.D()

    coordinates = mesh.node_coordinates()

    element_set = ElementSet(
        make_reference_element(mesh.element_type),
        coordinates,
        mesh.connectivity_array(),
    )

    elements = []

    for e in range(mesh.number_of_elements()):
        elements.append(PlaneElement(element_set, e, material))

    number_of_dofs = mesh.number_of_dofs()
    K = DenseMatrix(number_of_dofs)

    for element in elements:
        K.scatter_local_to_global(element)

    f = numpy.zeros(number_of_dofs)

    # Solucion exacta en todos los nodos: [u0, v0, u1, v1, ...]
    u_exact = numpy.zeros(number_of_dofs)

    for n in range(mesh.number_of_nodes()):
        x, y = coordinates[n]
        u_exact[2 * n], u_exact[2 * n + 1] = exact_displacement(x, y)

    # Desplazamiento impuesto en los nodos del borde del cuadrado
    number_of_interior_nodes = 0

    for n in range(mesh.number_of_nodes()):
        x, y = coordinates[n]

        on_boundary = (
            numpy.isclose(x, 0.0) or numpy.isclose(x, 3.0)
            or numpy.isclose(y, 0.0) or numpy.isclose(y, 3.0)
        )

        if on_boundary:
            for dof in mesh.node_global_dofs(n):
                apply_dirichlet_bc(K, f, dof, u_exact[dof])
        else:
            number_of_interior_nodes = number_of_interior_nodes + 1

    # Si no hubiera nodos interiores el test no verificaria nada
    assert number_of_interior_nodes > 0

    u = solve(K, f)

    displacement_error = numpy.max(numpy.abs(u - u_exact)) / numpy.max(numpy.abs(u_exact))

    # Tension en cada punto de Gauss: sigma = D B u_e
    sigma_exact = D @ numpy.array([A, B, C])
    stress_error = 0.0

    for element in elements:
        ue = u[element.global_dofs()]

        for g in range(element_set.number_of_points):
            sigma = D @ element.B_matrix(g) @ ue
            error = numpy.max(numpy.abs(sigma - sigma_exact)) / numpy.max(numpy.abs(sigma_exact))
            stress_error = max(stress_error, error)

    return displacement_error, stress_error


def check_patch(filename):
    for model in MODELS:
        displacement_error, stress_error = run_patch(filename, model)

        assert displacement_error < 1.0e-10
        assert stress_error < 1.0e-10


def test_patch_t3():
    check_patch("meshes/patch_t3.mesh")


def test_patch_q4():
    check_patch("meshes/patch_q4.mesh")


def test_patch_t6():
    check_patch("meshes/patch_t6.mesh")


def test_patch_q8():
    check_patch("meshes/patch_q8.mesh")


if __name__ == "__main__":
    tests = [
        test_patch_t3,
        test_patch_q4,
        test_patch_t6,
        test_patch_q8,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)

    print()
    print("Errores relativos (desplazamientos, tensiones):")

    for filename in MESHES:
        for model in MODELS:
            print(" ", filename, model, run_patch(filename, model))
