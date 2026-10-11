"""Convierte una malla de NX/Nastran (.dat) en archivos CSV.

Requiere pyNastran, pandas y numpy (ver read_mesh/requirements.txt).

Uso:

    python read_mesh/get_mesh_from_dat.py archivo.dat carpeta_de_salida prefijo
"""

import argparse
import os
import tempfile

from pyNastran.bdf.bdf import BDF
import pandas as pd
import numpy as np


def corrected_copy(bdf_file):
    """Devuelve la ruta de una copia temporal del archivo en la que la linea
    'BEGIN BULK' inicial esta comentada. El archivo original no se modifica.
    """
    with open(bdf_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if not line.startswith('$'):
            if line.strip().upper() == 'BEGIN BULK':
                lines[i] = '$' + line
            break

    handle, path = tempfile.mkstemp(suffix='.dat')

    with os.fdopen(handle, 'w') as f:
        f.writelines(lines)

    return path


def nastran_mesh_to_csv(bdf_file, output_dir, file_prefix,
                        export_nodes=True, export_elements=True,
                        export_properties=False, export_materials=False):
    temporary_file = corrected_copy(bdf_file)

    try:
        model = BDF()
        model.read_bdf(temporary_file, punch=True, xref=True, validate=True) # se leen todos los contenidos
    finally:
        os.remove(temporary_file)

    os.makedirs(output_dir, exist_ok=True)

    if export_nodes:
        nids = list(model.nodes.keys())
        xyz  = np.array([model.nodes[nid].xyz for nid in nids])
        df_nodes = pd.DataFrame({'nid': nids, 'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]})
        df_nodes.to_csv(os.path.join(output_dir, f'{file_prefix}_nodes.csv'), index=False)

    if export_elements:
        df_eles = pd.DataFrame(
            [{'eid': eid, 'type': elem.type, 'pid': elem.pid, 'nodes': elem.nodes} for eid, elem in model.elements.items()]
        )
        df_eles.to_csv(os.path.join(output_dir, f'{file_prefix}_elements.csv'), index=False)

    if export_properties:
        df_props = pd.DataFrame(
            [{'pid': pid, 'mid': prop.mid} for pid, prop in model.properties.items()]
        )
        df_props.to_csv(os.path.join(output_dir, f'{file_prefix}_properties.csv'), index=False)

    if export_materials:
        df_mats = pd.DataFrame(
            [{'mid': mid, 'E': mat.e, 'G': mat.g, 'nu': mat.nu} for mid, mat in model.materials.items()]
        )
        df_mats.to_csv(os.path.join(output_dir, f'{file_prefix}_materials.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Malla de NX/Nastran (.dat) a CSV")
    parser.add_argument("bdf_file", help="archivo .dat de entrada (no se modifica)")
    parser.add_argument("output_dir", help="carpeta donde se escriben los CSV")
    parser.add_argument("file_prefix", help="prefijo de los archivos de salida")
    arguments = parser.parse_args()

    nastran_mesh_to_csv(
        bdf_file=arguments.bdf_file,
        output_dir=arguments.output_dir,
        file_prefix=arguments.file_prefix,
        export_nodes=True, export_elements=True,
        export_properties=True, export_materials=True,
    )
    print('done')
