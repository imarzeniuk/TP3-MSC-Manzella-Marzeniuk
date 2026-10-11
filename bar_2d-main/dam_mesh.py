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

Uso, desde la carpeta que contiene a meshes/:

    python dam_mesh.py
"""

# Geometria (m)
H = 30.0        # altura total de la presa
B = 18.0        # ancho de la base
C = 3.0         # ancho de la cresta

# Cantidad de celdas
NX = 6
NY = 24

# Material (hormigon H21)
MODEL = "plane_strain"
E = 21.0e9
NU = 0.2
RHO = 2400.0
THICKNESS = 1.0

ORDER = {"T3": 1, "Q4": 1, "T6": 2, "Q8": 2}


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


def write_mesh(filename, kind, **geometry):
    nodes, elements = build_mesh(kind, **geometry)

    file = open(filename, "w")

    file.write("# Presa de gravedad, elementos " + kind + "\n")
    file.write("# n_nodes n_elements dim\n")
    file.write(str(len(nodes)) + " " + str(len(elements)) + " 2\n\n")

    file.write("# MATERIAL material_id model E nu rho thickness\n")
    file.write("MATERIAL 1 " + MODEL + " " + str(E) + " " + str(NU) + " "
               + str(RHO) + " " + str(THICKNESS) + "\n\n")

    file.write("# node_number x y\n")

    for number, (x, y) in enumerate(nodes, 1):
        file.write(str(number) + " " + format(x, ".4f") + " " + format(y, ".4f") + "\n")

    file.write("\n# element_number element_type order material_id connectivity\n")

    for number, connectivity in enumerate(elements, 1):
        file.write(str(number) + " " + kind + " " + str(ORDER[kind]) + " 1 "
                   + " ".join(str(n) for n in connectivity) + "\n")

    file.close()


if __name__ == "__main__":
    for kind in ORDER:
        write_mesh("meshes/dam_" + kind.lower() + ".mesh", kind)
        print("Escrita meshes/dam_" + kind.lower() + ".mesh")
