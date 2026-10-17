"""Generador de las mallas de la presa de gravedad (T3, T6, Q4 y Q8).

Perfil: paramento aguas arriba vertical (x = 0), cresta de ancho c arriba,
paramento aguas abajo recto hasta la base de ancho B:

    (0, H) ----- (c, H)
      |               \\
      |                 \\
      |                   \\
    (0, 0) ------------- (B, 0)

Se hace una grilla estructurada de nx x ny celdas: cada fila horizontal se
reparte entre x = 0 y el paramento aguas abajo. Los cuadrilateros salen de
cada celda; los triangulos, de partir cada celda en dos.

Para el estudio del ancho de la base se generan, ademas, mallas Q8 con
B = 24, 30 y 36 m (meshes/dam_q8_b24.mesh, etc.), con NX elegido para que los
elementos de la base sigan midiendo 3 m de ancho.

Uso, desde la carpeta que contiene a meshes/:

    python dam_mesh.py
"""

# Geometria (m)
H = 30.0        # altura total de la presa
B = 18.0        # ancho de la base
C = 3.0         # ancho de la cresta

multiplicador = 1

# Cantidad de celdas
NX = 6 * multiplicador
NY = 24 * multiplicador

# Anchos de base del estudio de geometria (m). Con B = 18 se generan las
# cuatro mallas de siempre; con los demas, solo Q8.
BASE_WIDTHS = [18.0, 24.0, 30.0, 36.0]

# Material (hormigon H21)
MODEL = "plane_strain"
E = 21.0e9
NU = 0.2
RHO = 2400.0
THICKNESS = 1.0

ORDER = {"T3": 1, "Q4": 1, "T6": 2, "Q8": 2}


def nx_for_base(base):
    """NX para un ancho de base: 6 elementos para B = 18 m y 2 mas cada 6 m
    (elementos de 3 m de ancho en la base).
    """
    return NX + 2 * round((base - B) / 6.0)


def mesh_filename(kind, base):
    """Nombre del archivo: meshes/dam_q8.mesh para B = 18 (los de siempre) y
    meshes/dam_q8_b24.mesh para los demas anchos.
    """
    name = "meshes/dam_" + kind.lower()

    if base != B:
        name = name + "_b" + format(base, "g")

    return name + ".mesh"


def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def grid_vertices(nx, ny, base, crest, height):
    """Diccionario (i, j) -> (x, y) con los vertices de la grilla.

    i va de 0 a nx (horizontal) y j de 0 a ny (vertical).
    """
    vertices = {}

    for j in range(ny + 1):
        y = height * j / ny

        # Ancho de la presa a la altura y: de B (base) a c (cresta)
        width = base + (crest - base) * y / height

        for i in range(nx + 1):
            vertices[(i, j)] = (width * i / nx, y)

    return vertices


def build_mesh(kind, nx=NX, ny=NY, base=B, crest=C, height=H):
    """Devuelve (nodos, elementos) del tipo pedido, con numeracion desde 1.

    nodos es una lista de (x, y); elementos es una lista de listas de nodos.
    """
    vertices = grid_vertices(nx, ny, base, crest, height)

    nodes = []
    index = {}

    def add(point):
        # Un nodo compartido por varios elementos se guarda una sola vez
        key = (round(point[0], 9), round(point[1], 9))

        if key not in index:
            nodes.append(point)
            index[key] = len(nodes)

        return index[key]

    elements = []

    for j in range(ny):
        for i in range(nx):
            p0 = vertices[(i, j)]
            p1 = vertices[(i + 1, j)]
            p2 = vertices[(i + 1, j + 1)]
            p3 = vertices[(i, j + 1)]

            if kind == "Q4":
                elements.append([add(p) for p in (p0, p1, p2, p3)])

            elif kind == "Q8":
                elements.append([add(p) for p in (
                    p0, p1, p2, p3,
                    midpoint(p0, p1), midpoint(p1, p2), midpoint(p2, p3), midpoint(p3, p0),
                )])

            else:
                for a, b, c in ((p0, p1, p2), (p0, p2, p3)):
                    if kind == "T3":
                        elements.append([add(a), add(b), add(c)])
                    else:
                        elements.append([
                            add(a), add(b), add(c),
                            add(midpoint(a, b)), add(midpoint(b, c)), add(midpoint(c, a)),
                        ])

    return nodes, elements


def write_mesh(filename, kind, base=B):
    nx = nx_for_base(base)
    nodes, elements = build_mesh(kind, nx=nx, base=base)

    file = open(filename, "w")

    file.write("# Presa de gravedad, elementos " + kind + ", B = " + format(base, "g")
               + " m, NX = " + str(nx) + ", NY = " + str(NY) + "\n")
    file.write("# n_nodes n_elements dim\n")
    file.write(str(len(nodes)) + " " + str(len(elements)) + " 2\n\n")

    file.write("# MATERIAL material_id model E nu rho thickness\n")
    file.write("MATERIAL 1 " + MODEL + " " + str(E) + " " + str(NU) + " "
               + str(RHO) + " " + str(THICKNESS) + "\n\n")

    file.write("# node_number x y\n")

    for number, (x, y) in enumerate(nodes, 1):
        # repr escribe todas las cifras del numero: asi los nodos de mitad de lado
        # no se corren del punto medio por redondeo
        file.write(str(number) + " " + repr(x) + " " + repr(y) + "\n")

    file.write("\n# element_number element_type order material_id connectivity\n")

    for number, connectivity in enumerate(elements, 1):
        file.write(str(number) + " " + kind + " " + str(ORDER[kind]) + " 1 "
                   + " ".join(str(n) for n in connectivity) + "\n")

    file.close()


if __name__ == "__main__":
    for base in BASE_WIDTHS:
        # Con la base actual, los cuatro tipos de elemento; con las demas, solo Q8
        if base == B:
            kinds = list(ORDER)
        else:
            kinds = ["Q8"]

        for kind in kinds:
            filename = mesh_filename(kind, base)
            write_mesh(filename, kind, base)
            print("Escrita", filename)
