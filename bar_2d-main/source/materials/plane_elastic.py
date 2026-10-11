import numpy


class PlaneElastic:
    """Material elastico lineal isotropo para problemas 2D.

    model = "plane_stress" (tension plana) o "plane_strain" (deformacion plana).
    El orden de deformaciones y tensiones es [x, y, xy] (con gamma_xy de
    ingenieria), por lo que sigma = D @ epsilon con D de 3 x 3.
    """

    def __init__(self, E, nu, model, rho=0.0, thickness=1.0):
        self.E = E
        self.nu = nu
        self.model = model
        self.rho = rho
        self.thickness = thickness

    @classmethod
    def from_spec(cls, spec):
        """Crea el material desde un MaterialSpec leido de la malla."""
        return cls(spec.E, spec.nu, spec.model, spec.rho, spec.thickness)

    def D(self):
        """Matriz constitutiva de 3 x 3."""
        E = self.E
        nu = self.nu

        if self.model == "plane_stress":
            factor = E / (1.0 - nu * nu)

            return factor * numpy.array([
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ])

        factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))

        return factor * numpy.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)],
        ])
