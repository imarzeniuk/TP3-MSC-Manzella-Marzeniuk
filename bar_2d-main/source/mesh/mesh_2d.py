import numpy

from source.mesh.node import Node


class MaterialSpec:
    """Datos de un material tal como se leen del archivo de malla.

    No calcula nada: solo guarda los numeros. La matriz D se arma aparte.
    """

    def __init__(self, global_id, model, E, nu, rho, thickness):
        self.global_id = global_id
        self.model = model
        self.E = E
        self.nu = nu
        self.rho = rho
        self.thickness = thickness


class Mesh2D:
    """Malla 2D con un unico tipo de elemento (T3, T6, Q4 o Q8).

    Formato del archivo (texto plano, '#' inicia un comentario, numeracion
    desde 1):

        n_nodes n_elements dim
        MATERIAL material_id model E nu rho thickness      (una o mas lineas)
        node_number x y                                    (n_nodes lineas)
        element_number element_type order material_id n1 n2 ...

    Internamente todo se guarda con numeracion desde 0, como en Python.
    """

    def __init__(self):
        self.dim = 2
        self.dofs_per_node = 2
        self.nodes = []
        self.materials = {}
        self.element_type = None
        self.element_order = None
        self.element_global_ids = []
        self.element_material_ids = []
        self.connectivity = []

    def read(self, filename):
        file = open(filename, "r")

        lines = []
        material_lines = []

        for raw_line in file:
            # Se descarta lo que hay despues de '#' y los espacios sobrantes
            line = raw_line.split("#")[0].strip()

            if line == "":
                continue

            if line.split()[0] == "MATERIAL":
                material_lines.append(line)
            else:
                lines.append(line)

        file.close()

        header = lines[0].split()

        n_nodes = int(header[0])
        n_elements = int(header[1])
        self.dim = int(header[2])

        for line in material_lines:
            values = line.split()

            global_id = int(values[1])

            self.materials[global_id] = MaterialSpec(
                global_id, values[2],
                float(values[3]), float(values[4]), float(values[5]), float(values[6]),
            )

        line_number = 1

        for i in range(n_nodes):
            values = lines[line_number].split()
            line_number = line_number + 1

            global_id = int(values[0]) - 1

            if global_id != i:
                raise ValueError("Unexpected node numbering")

            self.nodes.append(Node(global_id, float(values[1]), float(values[2])))

        for i in range(n_elements):
            values = lines[line_number].split()
            line_number = line_number + 1

            # Todos los elementos son del mismo tipo: se toma el de la primera linea
            self.element_type = values[1]
            self.element_order = int(values[2])

            connectivity = []

            for j in range(4, len(values)):
                connectivity.append(int(values[j]) - 1)

            self.element_global_ids.append(int(values[0]) - 1)
            self.element_material_ids.append(int(values[3]))
            self.connectivity.append(connectivity)

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_elements(self):
        return len(self.connectivity)

    def node(self, node_global_id):
        return self.nodes[node_global_id]

    def material(self, element_global_id):
        return self.materials[self.element_material_ids[element_global_id]]

    def element_node_ids(self, element_global_id):
        return self.connectivity[element_global_id]

    def element_coordinates(self, element_global_id):
        """Coordenadas de los nodos del elemento: matriz de n_nodos x 2."""
        ids = self.connectivity[element_global_id]

        coordinates = numpy.zeros((len(ids), 2))

        for i in range(len(ids)):
            coordinates[i, 0] = self.nodes[ids[i]].x
            coordinates[i, 1] = self.nodes[ids[i]].y

        return coordinates

    def node_coordinates(self):
        """Coordenadas de todos los nodos: matriz de n_nodos x 2 con [x, y]."""
        coordinates = numpy.zeros((self.number_of_nodes(), 2))

        for i in range(self.number_of_nodes()):
            coordinates[i, 0] = self.nodes[i].x
            coordinates[i, 1] = self.nodes[i].y

        return coordinates

    def connectivity_array(self):
        """Conectividad como matriz entera de n_elementos x nodos_por_elemento,
        con numeros globales de nodo en base 0 (se usa para indexar arrays).
        """
        return numpy.array(self.connectivity, dtype=int)

    def number_of_dofs(self):
        return self.number_of_nodes() * self.dofs_per_node

    def node_global_dofs(self, node_global_id):
        dofs = []

        for i in range(self.dofs_per_node):
            dofs.append(node_global_id * self.dofs_per_node + i)

        return dofs
