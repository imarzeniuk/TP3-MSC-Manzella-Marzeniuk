# TP3 – Mecánica del Sólido Computacional (ITBA)

Solver de elementos finitos en Python para elasticidad lineal plana (tensión plana o
deformación plana), aplicado a una presa de gravedad.

Autores: Tomás Manzella e Iván Marzeniuk.

## Elementos

| Nombre | Familia | Orden | Nodos | Cuadratura completa por defecto |
|---|---|---|---|---|
| `CST` (alias `T3`) | triángulo | 1 | 3 | 1 punto |
| `LST` (alias `T6`) | triángulo | 2 | 6 | 3 puntos |
| `Q4` | cuadrilátero | 1 | 4 | Gauss 2×2 |
| `Q8` | cuadrilátero | 2 | 8 | Gauss 3×3 |
| `L2`, `L3` | línea | 1, 2 | 2, 3 | Gauss 2, 3 (cargas sobre los lados) |

Numeración local: vértices en sentido antihorario y después los nodos de mitad de lado
(igual que VTK y Gmsh).

## Estructura

```
source/
  reference_elements/  Tria, Quad, Line y make_reference_element(nombre)
  quadrature/          Quadrature(familia, regla): Gauss-Legendre, reglas de triángulo
  elements/            ElementSet (N, dN/dx, w·detJ) y PlaneElement (B, Ke)
  materials/           PlaneElastic (matriz D) y check_material
  mesh/                Mesh2D: lector del formato .mesh
  systems/             numeración de DOFs, matriz global, condiciones de borde
  solvers/             resolución del sistema lineal
  postprocess/         resultantes de reacciones, control de la base y estática a mano
  output/              escritura de .vtu para ParaView
tests/                 un archivo de tests por módulo, más el patch test
meshes/                mallas de los patch tests y de la presa
dam_mesh.py            generador de las mallas de la presa
dam.py                 presa con la base empotrada: solución, reacciones y .vtu
read_mesh/             convertidor de mallas de NX/Nastran (en desarrollo)
```

## Formato de malla (`.mesh`)

Texto plano, `#` inicia un comentario y la numeración empieza en 1:

```
n_nodes n_elements dim
MATERIAL material_id model E nu rho thickness      # model: plane_stress | plane_strain
node_number x y                                    # n_nodes líneas
element_number element_type order material_id n1 n2 ...
```

El lector valida tipo y orden del elemento, cantidad de nodos, material existente y con
valores físicos (E > 0, −1 < ν < 0,5, ρ ≥ 0, espesor > 0), nodos en rango y orientación
antihoraria. Una malla tiene un solo tipo de elemento.

## Uso

Instalar dependencias (`numpy`, `meshio`):

```
python -m pip install -r requirements.txt
```

Correr los tests desde esta carpeta, uno por módulo:

```
python -m tests.test_patch
python -m tests.test_element_set
```

o todos juntos con `pytest` si está instalado. Regenerar las mallas de la presa:

```
python dam_mesh.py
```

## Presa con la base empotrada

```
python dam.py                                  # usa meshes/dam_q8.mesh
python dam.py meshes/dam_t3.mesh --scale 5000 --mu 0.7 --freeboard 2
```

Fija u = v = 0 en todos los nodos de la base (y = 0), aplica el peso propio y la presión
hidrostática, resuelve y calcula las reacciones `R = K u − f`. Imprime:

- la suma de reacciones y su momento respecto del pie, comparados con la estática a mano
  (peso, empuje, momento estabilizante y desestabilizante, factor de seguridad al vuelco y
  rozamiento mínimo);
- el equilibrio global entre reacciones y cargas;
- una tabla con las reacciones de cada nodo de la base, marcando los nodos traccionados o
  con `|R_x| > μ R_y`, donde el empotramiento deja de equivaler a un apoyo con rozamiento.

Al final escribe `results/<malla>.vtu` con la configuración deformada (`--scale` es el factor
de escala), para abrir en ParaView.
