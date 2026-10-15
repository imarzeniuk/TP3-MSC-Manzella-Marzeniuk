"""Presa de gravedad apoyada en la base.

Lee una malla, arma K y f (peso propio + presion hidrostatica), fija u = v = 0
en los nodos de la base (y = 0), resuelve, calcula las reacciones y las
compara con la estatica a mano. Despues calcula las tensiones y escribe un
.vtu con la configuracion deformada y los campos de tension.

Liberacion manual de la base: los nodos de la base que quedan traccionados en
realidad se despegan del lecho (traccion nula). Se puede elegir cuantos nodos
liberar, contados desde el talon (x = 0); el usuario va probando: si aparece
otro nodo apoyado traccionado, vuelve a correr liberando uno mas.

Prueba de la deformada (--uniform-test): sin peso propio y con una presion
uniforme sobre el paramento aguas arriba, con la misma resultante que la
hidrostatica. La presa debe curvarse como una mensula empotrada cargada en +x.

Uso, desde la carpeta que contiene a source/ y meshes/:

    python dam.py

El programa pregunta por consola el tipo de elemento, el factor de escala y
cuantos nodos de la base liberar. Tambien se pueden pasar por la linea de
comandos, y entonces no los pregunta:

    python dam.py meshes/dam_t3.mesh --scale 5000 --mu 0.7 --release 1
    python dam.py meshes/dam_q8.mesh --uniform-test --scale 2000
"""

import argparse

import numpy

from source.mesh.mesh_2d import Mesh2D
from source.materials.plane_elastic import PlaneElastic
from source.elements.element_set import ElementSet
from source.elements.plane_element import PlaneElement
from source.systems.dense_matrix import DenseMatrix
from source.systems.loads import (
    body_force_vector, hydrostatic_vector, uniform_pressure_vector,
)
from source.systems.boundary_conditions import (
    nodes_on_line, fixed_dofs, solve_with_dirichlet,
)
from source.postprocess.reactions import resultant, base_contact_table, dam_statics
from source.postprocess.stresses import (
    STRESS_NAMES, gauss_point_stresses, gauss_point_coordinates,
    principal_stresses, element_average, nodal_stresses,
)
from source.output.paraview_2d import write_results_2d


G = 9.81                # m/s2
WATER_DENSITY = 1000.0  # kg/m3


# Malla de la presa para cada nombre de elemento que se puede escribir en consola
DAM_MESHES = {
    "T3": "meshes/dam_t3.mesh",
    "CST": "meshes/dam_t3.mesh",
    "T6": "meshes/dam_t6.mesh",
    "LST": "meshes/dam_t6.mesh",
    "Q4": "meshes/dam_q4.mesh",
    "Q8": "meshes/dam_q8.mesh",
}

DEFAULT_ELEMENT = "Q8"
DEFAULT_SCALE = 1000.0
DEFAULT_RELEASED_NODES = 0     # nodos de la base liberados, contados desde el talon
RELEASE_MODE = "both"          # "both": libera u y v (traccion nula); "vertical": solo v


def ask_mesh():
    """Pregunta por consola el tipo de elemento, hasta recibir uno valido, y
    devuelve el archivo de malla que le corresponde. Enter usa el valor por defecto.
    """
    while True:
        # input muestra el texto y devuelve lo que se escribe, como texto
        answer = input(
            "Elemento (T3/CST, T6/LST, Q4, Q8) [" + DEFAULT_ELEMENT + "]: "
        )

        # strip saca los espacios y upper pasa a mayusculas: " q8 " -> "Q8"
        name = answer.strip().upper()

        if name == "":
            name = DEFAULT_ELEMENT

        if name in DAM_MESHES:
            return DAM_MESHES[name]

        print("  Opcion no valida. Escribir T3, CST, T6, LST, Q4 o Q8.")


def ask_scale():
    """Pregunta por consola el factor de escala de la deformada, hasta recibir
    un numero positivo. Enter usa el valor por defecto.
    """
    while True:
        answer = input("Factor de escala [" + str(DEFAULT_SCALE) + "]: ").strip()

        if answer == "":
            return DEFAULT_SCALE

        try:
            # Se acepta la coma decimal: "2,5" -> 2.5
            scale = float(answer.replace(",", "."))
        except ValueError:
            print("  Valor no valido. Escribir un numero, por ejemplo 1000.")
            continue

        if scale > 0.0:
            return scale

        print("  El factor de escala debe ser mayor que 0.")


def sorted_base_nodes(coordinates):
    """Nodos de la base (y = 0) ordenados por x, del talon al pie."""
    base_nodes = nodes_on_line(coordinates, axis=1, value=0.0)

    return sorted(base_nodes, key=lambda node_id: coordinates[node_id][0])


def ask_released_nodes(number_of_base_nodes):
    """Pregunta cuantos nodos de la base liberar, contados desde el talon (x = 0).
    Acepta un entero entre 0 y number_of_base_nodes - 2 (deben quedar al menos 2
    apoyados para impedir los movimientos rigidos). Enter usa el valor por defecto.
    """
    maximum = number_of_base_nodes - 2

    while True:
        answer = input(
            "Nodos de la base a liberar desde el talon (0 a " + str(maximum) + ") ["
            + str(DEFAULT_RELEASED_NODES) + "]: "
        ).strip()

        if answer == "":
            return DEFAULT_RELEASED_NODES

        try:
            released_nodes = int(answer)
        except ValueError:
            print("  Valor no valido. Escribir un numero entero, por ejemplo 1.")
            continue

        if 0 <= released_nodes <= maximum:
            return released_nodes

        print("  Debe estar entre 0 y " + str(maximum) + ".")


def solve_dam(mesh, water_level, released_nodes=0, release_mode="both",
              uniform_test=False):
    """Resuelve la presa apoyada en la base.

    released_nodes: cantidad de nodos de la base liberados, contados desde el
                    talon (x = 0). El resto queda con u = v = 0.
    release_mode:   "both" libera u y v de esos nodos (traccion nula);
                    "vertical" libera solo v y mantiene u = 0.
    uniform_test:   True para la prueba de la deformada: sin peso propio y con
                    una presion uniforme P = rho_w g h / 2 hasta el nivel del
                    agua (misma resultante que la hidrostatica).

    Devuelve (u, R, f, base_nodes, released, elements): desplazamientos,
    reacciones, cargas, nodos de la base ordenados por x, nodos liberados y la
    lista de PlaneElement (para calcular tensiones).
    """
    if release_mode not in ("both", "vertical"):
        raise ValueError("Unknown release mode: " + str(release_mode))

    coordinates = mesh.node_coordinates()

    base_nodes = sorted_base_nodes(coordinates)

    if not 0 <= released_nodes <= len(base_nodes) - 2:
        raise ValueError(
            "Expected between 0 and " + str(len(base_nodes) - 2) + " released nodes"
        )

    element_set = ElementSet(
        mesh.reference_element(), coordinates, mesh.connectivity_array()
    )

    materials = []

    for e in range(mesh.number_of_elements()):
        materials.append(PlaneElastic.from_spec(mesh.material(e)))

    K = DenseMatrix(mesh.number_of_dofs())

    elements = []

    for e in range(mesh.number_of_elements()):
        elements.append(PlaneElement(element_set, e, materials[e]))
        K.scatter_local_to_global(elements[e])

    if uniform_test:
        uniform_pressure = 0.5 * WATER_DENSITY * G * water_level

        f = uniform_pressure_vector(element_set, materials, uniform_pressure, water_level)
    else:
        f = (
            body_force_vector(element_set, materials, G)
            + hydrostatic_vector(element_set, materials, WATER_DENSITY, G, water_level)
        )

    # Los primeros released_nodes de la base (desde el talon) se liberan; el
    # resto queda con u = v = 0
    released = base_nodes[:released_nodes]
    supported = base_nodes[released_nodes:]

    prescribed_dofs = fixed_dofs(supported)

    if release_mode == "vertical":
        # Solo se libera v: los liberados mantienen u = 0
        prescribed_dofs = sorted(prescribed_dofs + fixed_dofs(released, components=(0,)))

    prescribed_values = numpy.zeros(len(prescribed_dofs))

    u, R = solve_with_dirichlet(K.values, f, prescribed_dofs, prescribed_values)

    return u, R, f, base_nodes, released, elements


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
    parser.add_argument("mesh", nargs="?", default=None,
                        help="archivo de malla (si no se indica, se pregunta el elemento)")
    parser.add_argument("--freeboard", type=float, default=2.0,
                        help="borde libre en m: el agua llega hasta H - borde libre")
    parser.add_argument("--mu", type=float, default=0.6,
                        help="coeficiente de rozamiento para el control de la base")
    parser.add_argument("--scale", type=float, default=None,
                        help="factor de escala de la deformada (si no se indica, se pregunta)")
    parser.add_argument("--output", default=None,
                        help="archivo .vtu de salida (por defecto results/<malla>.vtu)")
    parser.add_argument("--release", type=int, default=None,
                        help="nodos de la base a liberar, contados desde el talon "
                             "(si no se indica, se pregunta)")
    parser.add_argument("--uniform-test", action="store_true",
                        help="prueba de la deformada: sin peso propio y con presion "
                             "uniforme sobre el paramento (base empotrada)")
    arguments = parser.parse_args()

    # Lo que no vino por la linea de comandos se pregunta por consola
    if arguments.mesh is None:
        arguments.mesh = ask_mesh()

    if arguments.scale is None:
        arguments.scale = ask_scale()

    mesh = Mesh2D()
    mesh.read(arguments.mesh)

    coordinates = mesh.node_coordinates()
    material = mesh.material(0)

    B, c, H = dam_dimensions(coordinates)
    water_level = H - arguments.freeboard

    base_nodes = sorted_base_nodes(coordinates)

    if arguments.uniform_test:
        # En la prueba de la deformada la base queda empotrada
        if arguments.release is not None and arguments.release != 0:
            print("Con --uniform-test la base queda empotrada: se ignora --release.")

        arguments.release = 0

    elif arguments.release is None:
        print()
        print("La base tiene", len(base_nodes), "nodos, ordenados desde el talon (x en m):")
        print("  " + ", ".join(format(coordinates[n][0], "g") for n in base_nodes))

        arguments.release = ask_released_nodes(len(base_nodes))

    print()

    u, R, f, base_nodes, released, elements = solve_dam(
        mesh, water_level, arguments.release, RELEASE_MODE, arguments.uniform_test
    )

    all_nodes = range(mesh.number_of_nodes())
    toe = (B, 0.0)

    # Reacciones de la base y cargas totales, con momentos respecto del pie
    Rx, Ry, M_R = resultant(coordinates, R, base_nodes, toe)
    fx, fy, M_f = resultant(coordinates, f, all_nodes, toe)

    print("Malla:", arguments.mesh, "(" + mesh.element_type + ",",
          mesh.number_of_nodes(), "nodos,", mesh.number_of_elements(), "elementos)")
    print("Geometria: B =", B, "m, c =", c, "m, H =", H, "m, agua hasta", water_level, "m")

    if len(released) == 0:
        print("Base empotrada:", len(base_nodes), "nodos con u = v = 0")
    else:
        if RELEASE_MODE == "both":
            what = "u y v libres (traccion nula)"
        else:
            what = "v libre, u = 0"

        print("Base:", len(base_nodes) - len(released), "nodos apoyados con u = v = 0 y",
              len(released), "liberados desde el talon (" + what + ")")

    print()

    if arguments.uniform_test:
        print("PRUEBA DE LA DEFORMADA: sin peso propio y con presion uniforme")
        print("  P = rho_w g h / 2 = {:.0f} Pa sobre x = 0, hasta y = {} m".format(
            0.5 * WATER_DENSITY * G * water_level, water_level))
        print("  Suma F_x de la carga: {:.4f} MN (igual al empuje hidrostatico)".format(
            fx / 1.0e6))
        print("  La comparacion con la estatica a mano no vale para esta prueba.")
        print()

        print("Reacciones de la base (FEM)")
        print("  Suma R_x:             {:>12.4f} MN".format(Rx / 1.0e6))
        print("  Suma R_y:             {:>12.4f} MN".format(Ry / 1.0e6))
        print("  M_R respecto del pie: {:>12.4f} MN m".format(M_R / 1.0e6))
        print()

    else:
        statics = dam_statics(
            B, c, H, water_level, material.rho, WATER_DENSITY, G, material.thickness
        )

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

    if not arguments.uniform_test:
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

    # Vertice superior derecho de la cresta: el nodo mas cercano a (c, H)
    crest = numpy.argmin(numpy.sum((coordinates - [c, H]) ** 2, axis=1))
    print("Cresta, vertice derecho ({:g}, {:g}): u = {:.4f} mm, v = {:.4f} mm".format(
        c, H, 1000.0 * u[2 * crest], 1000.0 * u[2 * crest + 1]))
    print()

    table = base_contact_table(coordinates, R, base_nodes, arguments.mu)

    print("Control de contacto en la base (mu = " + str(arguments.mu) + ")")
    print("  {:>8}{:>12}{:>12}{:>12}{:>10}{:>10}  {}".format(
        "x (m)", "R_x (kN)", "R_y (kN)", "|R_x|/R_y", "u (mm)", "v (mm)", ""))

    number_of_tension = 0
    number_of_sliding = 0
    number_of_penetrating = 0

    for row in table:
        node_id = row["node"]
        # Desplazamientos del nodo en mm (en los apoyados valen 0)
        u_node = 1000.0 * u[2 * node_id]
        v = 1000.0 * u[2 * node_id + 1]
        flags = []

        if node_id in released:
            # Nodo despegado: no tiene sentido controlar traccion ni rozamiento
            flags.append("LIBERADO")

            if v < 0.0:
                flags.append("PENETRA (liberar uno menos)")
                number_of_penetrating = number_of_penetrating + 1

            print("  {:>8.3f}{:>12.1f}{:>12}{:>12}{:>10.4f}{:>10.4f}  {}".format(
                row["x"], row["Rx"] / 1.0e3, "-", "-", u_node, v, " ".join(flags)))
            continue

        if row["tension"]:
            flags.append("TRACCION")
            number_of_tension = number_of_tension + 1

        if row["sliding"]:
            flags.append("DESLIZA")
            number_of_sliding = number_of_sliding + 1

        print("  {:>8.3f}{:>12.1f}{:>12.1f}{:>12.3f}{:>10}{:>10}  {}".format(
            row["x"], row["Rx"] / 1.0e3, row["Ry"] / 1.0e3, row["ratio"], "-", "-",
            " ".join(flags)))

    print()

    if arguments.uniform_test:
        # Sin peso propio nada comprime la base: la traccion es esperable y en
        # esta prueba no se liberan nodos
        print("Prueba uniforme sin peso: la base queda traccionada del lado del agua,")
        print("como se espera sin peso propio. En esta prueba no se liberan nodos.")
    elif number_of_tension > 0:
        print("Hay nodos apoyados traccionados: probar liberando",
              len(released) + 1, "(--release " + str(len(released) + 1) + ").")
    else:
        print("Ningun nodo apoyado traccionado.")

    if number_of_penetrating > 0:
        print("Hay nodos liberados que penetran el lecho: probar liberando",
              len(released) - 1, "(--release " + str(len(released) - 1) + ").")

    if number_of_sliding > 0 and not arguments.uniform_test:
        print(number_of_sliding, "nodos apoyados con |R_x| > mu R_y: ahi un apoyo con")
        print("rozamiento no podria dar la fuerza horizontal del modelo.")

    print()

    # Tensiones en los puntos de Gauss: (elementos, puntos de Gauss, 4)
    element_set = elements[0].element_set

    sigma_gp = gauss_point_stresses(elements, u)
    principal_gp = principal_stresses(sigma_gp)
    gp_coordinates = gauss_point_coordinates(element_set)

    # numpy.argmax devuelve la posicion en el arreglo "aplanado";
    # unravel_index la convierte en (elemento, punto de Gauss)
    e_max, g_max = numpy.unravel_index(
        numpy.argmax(principal_gp["sigma_max"]), principal_gp["sigma_max"].shape)
    e_min, g_min = numpy.unravel_index(
        numpy.argmin(principal_gp["sigma_min"]), principal_gp["sigma_min"].shape)

    print("Tensiones principales en los puntos de Gauss (traccion positiva)")
    print("  sigma_max maxima: {:>9.4f} MPa en ({:.3f}, {:.3f}) m, elemento {}".format(
        principal_gp["sigma_max"][e_max, g_max] / 1.0e6,
        gp_coordinates[e_max, g_max, 0], gp_coordinates[e_max, g_max, 1], e_max + 1))
    print("  sigma_min minima: {:>9.4f} MPa en ({:.3f}, {:.3f}) m, elemento {}".format(
        principal_gp["sigma_min"][e_min, g_min] / 1.0e6,
        gp_coordinates[e_min, g_min, 0], gp_coordinates[e_min, g_min, 1], e_min + 1))
    print()

    if len(released) == 0:
        print("  Advertencia: el maximo de traccion cae en el talon (0, 0), donde la base")
        print("  empotrada se encuentra con el paramento cargado. Es una singularidad del")
        print("  modelo y crece al refinar la malla: no es un valor de diseno.")
    else:
        print("  Advertencia: al liberar nodos, el maximo de traccion se corre al borde de")
        print("  la zona de contacto. Ese borde sigue siendo una concentracion de tensiones")
        print("  que depende de la malla.")

    print()

    # Valores en los nodos: se extrapolan las componentes y despues se calculan
    # las principales
    sigma_nodes = nodal_stresses(element_set, sigma_gp)
    principal_nodes = principal_stresses(sigma_nodes)

    point_data = {}

    for i in range(len(STRESS_NAMES)):
        point_data[STRESS_NAMES[i]] = sigma_nodes[:, i]

    for name in ("sigma_1", "sigma_2", "sigma_max", "sigma_min"):
        point_data[name] = principal_nodes[name]

    # Valores por elemento: promedio de las componentes y sigma_max del promedio
    sigma_elements = element_average(element_set, sigma_gp)

    cell_data = {}

    for i in range(len(STRESS_NAMES)):
        cell_data[STRESS_NAMES[i]] = sigma_elements[:, i]

    cell_data["sigma_max"] = principal_stresses(sigma_elements)["sigma_max"]

    output = arguments.output

    if output is None:
        # "meshes/dam_q8.mesh" -> "results/dam_q8.vtu"
        name = arguments.mesh.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]

        # Sufijo segun la corrida, para no pisar resultados anteriores
        if arguments.uniform_test:
            name = name + "_uniforme_sin_peso"
        elif len(released) > 0:
            name = name + "_libera" + str(len(released))

        output = "results/" + name + ".vtu"

    write_results_2d(output, mesh, u, scale=arguments.scale,
                     point_data=point_data, cell_data=cell_data)

    print("Escrito", output, "(factor de escala " + str(arguments.scale) + ")")
    print("  campos en los nodos:    ", ", ".join(point_data))
    print("  campos en los elementos:", ", ".join(cell_data))
