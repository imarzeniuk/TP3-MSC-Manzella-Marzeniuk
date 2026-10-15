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
  postprocess/         reacciones y estática a mano; tensiones (puntos de Gauss, nodos, principales)
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

Genera las cuatro mallas de siempre con B = 18 m (`dam_t3`, `dam_t6`, `dam_q4`, `dam_q8`) y,
para el estudio del ancho de la base, mallas Q8 con B = 24, 30 y 36 m (`dam_q8_b24.mesh`,
etc.). NX crece 2 elementos cada 6 m de base, para que los elementos de la base sigan midiendo
3 m; NY queda en 24.

## Presa con la base empotrada

```
python dam.py
```

El programa pregunta por consola, en este orden:

1. el ancho de la base: 18, 24, 30 o 36 m;
2. el tipo de elemento (`T3`/`CST`, `T6`/`LST`, `Q4` o `Q8`), solo con B = 18 m: para los demás
   anchos hay solo malla Q8 y no se pregunta;
3. el factor de escala de la deformada;
4. cuántos nodos de la base liberar.

Con Enter se usan los valores por defecto (18 m, `Q8`, 1000 y 0). También se pueden pasar por
la línea de comandos, y entonces no se preguntan:

```
python dam.py meshes/dam_t3.mesh --scale 5000 --mu 0.7 --freeboard 2 --release 1
python dam.py --base 24 --scale 1000 --release 1
```

Si se pasa el archivo de malla, no se pregunta ni la base ni el elemento: B se mide sobre la
malla (y se ignora `--base`).

### Estudio del ancho de la base

Para aumentar el factor de seguridad se ensancha la base manteniendo el resto de la geometría
(paramento aguas arriba vertical, cresta de 3 m, altura de 30 m) y el nivel del agua (28 m):

```
python dam.py --base 18
python dam.py --base 24
python dam.py --base 30
python dam.py --base 36
```

Cada corrida compara las reacciones con la estática a mano de ese ancho (peso, momentos
respecto del pie, factor de seguridad al vuelco y rozamiento mínimo) y escribe
`results/dam_q8_b24.vtu`, etc. (con B = 18, `results/dam_q8.vtu`). Conviene liberar nodos del
talón hasta que no quede ningún apoyado traccionado: con las mallas actuales son 2 nodos con
B = 18, 1 con B = 24 y 30, y ninguno con B = 36.

### Liberación manual de la base

Donde la base queda traccionada, la presa en realidad se despega del lecho y ahí la condición
de borde es de tracción nula. `--release N` (o la pregunta por consola) libera los N primeros
nodos de la base contados desde el talón (x = 0); antes de preguntar se listan las x de los
nodos de la base (en LST y Q8 incluye los nodos de mitad de lado). El procedimiento es manual:

1. Correr con N = 0 (base empotrada) y mirar la tabla de contacto.
2. Si un nodo apoyado aparece como `TRACCION`, volver a correr con N + 1.
3. Si un nodo liberado aparece como `PENETRA` (v < 0, entra en el lecho rígido), se liberó de
   más: volver a N − 1.

Con las mallas actuales alcanza con N = 1 en CST y Q4 y con N = 2 en LST y Q8. Las resultantes
ΣR_x, ΣR_y y M_R no cambian al liberar nodos, porque las fija el equilibrio. Al principio de
`dam.py`, `RELEASE_MODE = "both"` libera u y v (tracción nula); con `"vertical"` se libera
solo v y se mantiene u = 0. El `.vtu` se guarda como `results/<malla>_liberaN.vtu`.

### Prueba de la deformada

```
python dam.py meshes/dam_q8.mesh --uniform-test --scale 2000
```

Sin peso propio y con una presión **uniforme** P = ρw g h / 2 sobre el paramento aguas arriba
hasta el nivel del agua (misma resultante que la hidrostática), con la base empotrada (se
ignora `--release`). La presa debe curvarse como una ménsula empotrada cargada en +x: en
ParaView, los incrementos de u_x del paramento crecen con la altura. La comparación con la estática a mano
no vale en esta prueba, así que solo se imprime el equilibrio. El `.vtu` se guarda como
`results/<malla>_uniforme_sin_peso.vtu`.

En el caso habitual (peso + agua) los incrementos de u_x decrecen con la altura: la sección es
asimétrica y el peso de la parte superior flexiona la presa hacia aguas arriba, lo que
compensa en parte la flexión por el agua.

Fija u = v = 0 en todos los nodos de la base (y = 0), aplica el peso propio y la presión
hidrostática, resuelve y calcula las reacciones `R = K u − f`. Imprime:

- la suma de reacciones y su momento respecto del pie, comparados con la estática a mano
  (peso, empuje, momento estabilizante y desestabilizante, factor de seguridad al vuelco y
  rozamiento mínimo);
- el equilibrio global entre reacciones y cargas;
- el desplazamiento de la cresta;
- una tabla con las reacciones de cada nodo de la base, marcando los apoyados traccionados o
  con `|R_x| > μ R_y` (donde el apoyo deja de equivaler a un contacto con rozamiento) y, para
  los liberados, sus desplazamientos u y v y si penetran el lecho;
- la máxima y la mínima tensión principal en los puntos de Gauss, con su ubicación y elemento.

Al final escribe `results/<malla>.vtu` con la configuración deformada (`--scale` es el factor
de escala), para abrir en ParaView. Campos del archivo, en Pa y con tracción positiva:

| Campo | Dónde | Contenido |
|---|---|---|
| `displacement` | nodos | desplazamiento real (sin escalar) |
| `u_x`, `u_y`, `u_magnitude` | nodos | componentes x e y del desplazamiento real y su módulo |
| `sigma_x`, `sigma_y`, `tau_xy`, `sigma_z` | nodos y elementos | componentes de tensión |
| `sigma_1`, `sigma_2` | nodos | tensiones principales en el plano |
| `sigma_max`, `sigma_min` | nodos (`sigma_max` también en elementos) | máxima y mínima tensión principal, incluida `sigma_z` |

Las tensiones se calculan en los puntos de Gauss (`σ = D B u`). Los valores de los nodos se
obtienen extrapolando las componentes desde los puntos de Gauss y promediando entre los
elementos que comparten el nodo; los de los elementos son el promedio ponderado por área.
Para la distribución de la máxima tensión principal sobre la deformada, colorear por
`sigma_max`.

El máximo de tracción cae en el talón (0, 0), donde la base empotrada se encuentra con el
paramento cargado: es una singularidad del modelo que crece al refinar la malla, no un valor
de diseño.
