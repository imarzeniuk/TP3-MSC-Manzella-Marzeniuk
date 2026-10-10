"""Verificaciones del material elastico plano.

Correr desde la carpeta que contiene a source/ y tests/:

    python -m tests.test_plane_elastic
"""

import numpy

from source.materials.plane_elastic import PlaneElastic
from source.mesh.mesh_2d import MaterialSpec


E = 21.0e9
NU = 0.25


def test_plane_stress_matrix():
    D = PlaneElastic(E, NU, "plane_stress").D()
    f = E / (1.0 - NU ** 2)

    expected = f * numpy.array([
        [1.0, NU, 0.0],
        [NU, 1.0, 0.0],
        [0.0, 0.0, (1.0 - NU) / 2.0],
    ])

    assert numpy.allclose(D, expected)


def test_plane_strain_matrix():
    D = PlaneElastic(E, NU, "plane_strain").D()
    f = E / ((1.0 + NU) * (1.0 - 2.0 * NU))

    expected = f * numpy.array([
        [1.0 - NU, NU, 0.0],
        [NU, 1.0 - NU, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * NU) / 2.0],
    ])

    assert numpy.allclose(D, expected)


def test_symmetric_positive_definite():
    for model in ("plane_stress", "plane_strain"):
        D = PlaneElastic(E, NU, model).D()

        assert numpy.allclose(D, D.T)
        assert numpy.all(numpy.linalg.eigvalsh(D) > 0.0)


def test_uniaxial_stress():
    # Traccion uniaxial en tension plana: eps = [e, -nu e, 0] da sigma = [E e, 0, 0]
    D = PlaneElastic(E, NU, "plane_stress").D()
    sigma = D @ numpy.array([1.0e-3, -NU * 1.0e-3, 0.0])

    assert numpy.allclose(sigma, [E * 1.0e-3, 0.0, 0.0])


def test_shear_modulus():
    # D[2, 2] es el modulo de corte G = E / (2 (1 + nu)) en ambos modelos
    G = E / (2.0 * (1.0 + NU))

    for model in ("plane_stress", "plane_strain"):
        assert numpy.isclose(PlaneElastic(E, NU, model).D()[2, 2], G)


def test_models_coincide_for_nu_zero():
    stress = PlaneElastic(E, 0.0, "plane_stress").D()
    strain = PlaneElastic(E, 0.0, "plane_strain").D()

    assert numpy.allclose(stress, strain)


def test_plane_strain_is_stiffer():
    # Con nu > 0 la deformacion plana restringe mas: D[0, 0] es mayor
    stress = PlaneElastic(E, NU, "plane_stress").D()
    strain = PlaneElastic(E, NU, "plane_strain").D()

    assert strain[0, 0] > stress[0, 0]


def test_from_spec():
    spec = MaterialSpec(1, "plane_strain", E, NU, 2400.0, 1.0)
    material = PlaneElastic.from_spec(spec)

    assert material.model == "plane_strain"
    assert material.rho == 2400.0
    assert numpy.allclose(material.D(), PlaneElastic(E, NU, "plane_strain").D())


if __name__ == "__main__":
    tests = [
        test_plane_stress_matrix,
        test_plane_strain_matrix,
        test_symmetric_positive_definite,
        test_uniaxial_stress,
        test_shear_modulus,
        test_models_coincide_for_nu_zero,
        test_plane_strain_is_stiffer,
        test_from_spec,
    ]

    for test in tests:
        test()
        print("OK ", test.__name__)
