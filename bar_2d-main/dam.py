"""Presa de gravedad con la base empotrada.

Lee una malla, arma K y f (peso propio + presion hidrostatica), fija u = v = 0
en todos los nodos de la base (y = 0), resuelve, calcula las reacciones y las
compara con la estatica a mano. Al final escribe un .vtu con la configuracion
deformada.

Uso, desde la carpeta que contiene a source/ y meshes/:

    python dam.py                                   (usa meshes/dam_q8.mesh)
    python dam.py meshes/dam_t3.mesh --scale 5000 --mu 0.7
"""

import argparse

import numpy

from source.mesh.mesh_2d import Mesh2D
from source.materials.plane_elastic import PlaneElastic
from source.elements.element_set import ElementSet
from source.elements.plane_element import PlaneElement
from source.systems.dense_matrix import DenseMatrix
from source.systems.loads import body_force_vector, hydrostatic_vector
from source.systems.boundary_conditions import (
    nodes_on_line, fixed_dofs, solve_with_dirichlet,
)
from source.postprocess.reactions import resultant, base_contact_table, dam_statics
from source.output.paraview_2d import write_results_2d


G = 9.81                # m/s2
WATER_DENSITY = 1000.0  # kg/m3


def solve_dam(mesh, water_level):
    """Resuelve la presa con la base empotrada. Devuelve (u, R, f, base_nodes):
    desplazamientos, reacciones, cargas y nodos de la base.
    """
    coordinates = mesh.node_coordinates()

    element_set = ElementSet(
        mesh.reference_element(), coordinates, mesh.connectivity_array()
    )

    materials = []

    for e in range(mesh.number_of_elements()):
        materials.append(PlaneElastic.from_spec(mesh.material(e)))

    K = DenseMatrix(mesh.number_of_dofs())

    for e in range(mesh.number_of_elements()):
        K.scatter_local_to_global(PlaneElement(element_set, e, materials[e]))

    f = (
        body_force_vector(element_set, materials, G)
        + hydrostatic_vector(element_set, materials, WATER_DENSITY, G, water_level)
    )

    # Base empotrada: u = v = 0 en todos los nodos con y = 0
    base_nodes = nodes_on_line(coordinates, axis=1, value=0.0)
    prescribed_dofs = fixed_dofs(base_nodes)
    prescribed_values = numpy.zeros(len(prescribed_dofs))

    u, R = solve_with_dirichlet(K.values, f, prescribed_dofs, prescribed_values)

    return u, R, f, base_nodes


def dam_dimensions(coordinates):
    """Base B, cresta c y altura H del perfil, medidos sobre la malla."""
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    H = y.max()
    tol = 1.0e-9 * H

    B = x[numpy.abs(y) <= tol].max()
    c = x[numpy.abs(y - H) <= tol].max()

    return B, c, H


def print_row(name, fem, hand, unit):
    # "{:>14.4f}": numero con 4 decimales alineado a la derecha en 14 lugares
    print("  {:<28}{:>14.4f}{:>14.4f}  {}".format(name, fem, hand, unit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Presa de gravedad con la base empotrada")
    parser.add_argument("mesh", nargs="?", default="meshes/dam_q8.mesh",
                        help="archivo de malla (por defecto meshes/dam_q8.mesh)")
    parser.add_argument("--freeboard", type=float, default=2.0,
                        help="borde libre en m: el agua llega hasta H - borde libre")
    parser.add_argument("--mu", type=float, default=0.6,
                        help="coeficiente de rozamiento para el control de la base")
    parser.add_argument("--scale", type=float, default=1000.0,
                        help="factor de escala de la deformada en el .vtu")
    parser.add_argument("--output", default=None,
                        help="archivo .vtu de salida (por defecto results/<malla>.vtu)")
    arguments = parser.parse_args()

    mesh = Mesh2D()
    mesh.read(arguments.mesh)

    coordinates = mesh.node_coordinates()
    material = mesh.material(0)

    B, c, H = dam_dimensions(coordinates)
    water_level = H - arguments.freeboard

    u, R, f, base_nodes = solve_dam(mesh, water_level)

    all_nodes = range(mesh.number_of_nodes())
    toe = (B, 0.0)

    # Reacciones de la base y cargas totales, con momentos respecto del pie
    Rx, Ry, M_R = resultant(coordinates, R, base_nodes, toe)
    fx, fy, M_f = resultant(coordinates, f, all_nodes, toe)

    statics = dam_statics(
        B, c, H, water_level, material.rho, WATER_DENSITY, G, material.thickness
    )

    print("Malla:", arguments.mesh, "(" + mesh.element_type + ",",
          mesh.number_of_nodes(), "nodos,", mesh.number_of_elements(), "elementos)")
    print("Geometria: B =", B, "m, c =", c, "m, H =", H, "m, agua hasta", water_level, "m")
    print("Base empotrada:", len(base_nodes), "nodos,", 2 * len(base_nodes), "DOFs fijos")
    print()

    print("Reacciones de la base (FEM) y estatica a mano, por metro de presa")
    print("  {:<28}{:>14}{:>14}".format("", "FEM", "a mano"))
    print_row("Suma R_x", Rx / 1.0e6, -statics["E"] / 1.0e6, "MN")
    print_row("Suma R_y", Ry / 1.0e6, statics["W"] / 1.0e6, "MN")
    print_row("M_R respecto del pie", M_R / 1.0e6,
              -(statics["M_W"] + statics["M_E"]) / 1.0e6, "MN m")
    print()

    print("Equilibrio global (deben dar 0): reacciones + cargas")
    print("  fuerza en x: {:.3e} N".format(Rx + fx))
    print("  fuerza en y: {:.3e} N".format(Ry + fy))
    print("  momento:     {:.3e} N m".format(M_R + M_f))
    print()

    print("Estatica a mano (momentos respecto del pie, antihorario positivo)")
    print("  M_W estabilizante:    {:>12.4f} MN m".format(statics["M_W"] / 1.0e6))
    print("  M_E desestabilizante: {:>12.4f} MN m".format(statics["M_E"] / 1.0e6))
    print("  FS al vuelco:         {:>12.4f}".format(statics["FS"]))
    print("  mu minimo (E / W):    {:>12.4f}".format(statics["mu_min"]))
    print("  resultante al pie:    {:>12.4f} m  (e = {:.4f} m, B/6 = {:.4f} m)".format(
        statics["x_r"], statics["e"], B / 6.0))
    print("  sigma base pie/talon: {:>12.4f} / {:.4f} MPa (compresion)".format(
        statics["sigma_toe"] / 1.0e6, statics["sigma_heel"] / 1.0e6))
    print()

    print("Desplazamiento maximo: u = {:.4f} mm, v = {:.4f} mm".format(
        1000.0 * u[0::2][numpy.argmax(numpy.abs(u[0::2]))],
        1000.0 * u[1::2][numpy.argmax(numpy.abs(u[1::2]))]))
    print()

    table = base_contact_table(coordinates, R, base_nodes, arguments.mu)

    print("Control de contacto en la base (mu = " + str(arguments.mu) + ")")
    print("  {:>8}{:>12}{:>12}{:>12}  {}".format("x (m)", "R_x (kN)", "R_y (kN)", "|R_x|/R_y", ""))

    number_of_flags = 0

    for row in table:
        flags = []

        if row["tension"]:
            flags.append("TRACCION")

        if row["sliding"]:
            flags.append("DESLIZA")

        if len(flags) > 0:
            number_of_flags = number_of_flags + 1

        print("  {:>8.3f}{:>12.1f}{:>12.1f}{:>12.3f}  {}".format(
            row["x"], row["Rx"] / 1.0e3, row["Ry"] / 1.0e3, row["ratio"], " ".join(flags)))

    print()

    if number_of_flags == 0:
        print("Ningun nodo traccionado ni con |R_x| > mu R_y: el empotramiento es")
        print("consistente con un contacto real con rozamiento.")
    else:
        print(number_of_flags, "nodos de la base no cumplen el contacto real: ahi el")
        print("empotramiento ejerce fuerzas que un apoyo con rozamiento no podria dar.")

    print()

    output = arguments.output

    if output is None:
        # "meshes/dam_q8.mesh" -> "results/dam_q8.vtu"
        name = arguments.mesh.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
        output = "results/" + name + ".vtu"

    write_results_2d(output, mesh, u, scale=arguments.scale)

    print("Escrito", output, "(factor de escala " + str(arguments.scale) + ")")
