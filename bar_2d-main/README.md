# TP3 – MSC - Presa

Elementos finitos 2D en Python (CST, LST, Q4, Q8) aplicado a una presa de gravedad.
Autores: Tomás Manzella e Iván Marzeniuk.

## Cómo correr la presa

El programa pregunta (Enter usa el valor entre corchetes):

1. Ancho de la base en m: 18, 24, 30 o 36 [18].
2. Elemento: T3/CST, T6/LST, Q4 o Q8 [Q8]. Solo se pregunta con base de 18 m; las otras bases tienen solo malla Q8.
3. Factor de escala de la deformada [1000].
4. Nodos de la base a liberar desde el talón, para los que quedan traccionados [0].

Resuelve con la base empotrada y después con la base elástica (Robin), imprime reacciones, momentos y tensiones, y escribe los resultados en `results/*.vtu` para abrir en ParaView.

## Estructura

dam.py          presa: resuelve y escribe los resultados
dam_mesh.py     genera las mallas de la presa
source/         elementos, cuadratura, malla, material, cargas, apoyos, tensiones y salida .vtu
tests/          tests de cada módulo y patch test
meshes/         mallas (patch test y presa)
results/        archivos .vtu
read_mesh/      convertidor de mallas de NX

