import numpy


PLANE_MODELS = ["plane_stress", "plane_strain"]


def check_material(model, E, nu, rho, thickness):
    """Verifica que los datos del material tengan sentido fisico; si no, lanza
    ValueError. La usan PlaneElastic y el lector de mallas.

    En un material isotropo estable -1 < nu < 0.5. Con nu = 0.5 (incompresible)
    la matriz de deformacion plana divide por (1 - 2 nu) = 0.
    """
    if model not in PLANE_MODELS:
        raise ValueError("Unknown material model: " + str(model))

    # Se escribe "not E > 0" en vez de "E <= 0" para rechazar tambien NaN
    if not E > 0.0:
        raise ValueError("Expected E > 0, got " + str(E))

    if not -1.0 < nu < 0.5:
        raise ValueError("Expected -1 < nu < 0.5, got " + str(nu))

    if not rho >= 0.0:
        raise ValueError("Expected rho >= 0, got " + str(rho))

    if not thickness > 0.0:
        raise ValueError("Expected thickness > 0, got " + str(thickness))


class PlaneElastic:
    """Material elastico lineal isotropo para problemas 2D.

    model = "plane_stress" (tension plana) o "plane_strain" (deformacion plana).
    El orden de deformaciones y tensiones es [x, y, xy] (con gamma_xy de
    ingenieria), por lo que sigma = D @ epsilon con D de 3 x 3.
    """

    def __init__(self, E, nu, model, rho=0.0, thickness=1.0):
        check_material(model, E, nu, rho, thickness)

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

    def out_of_plane_stress(self, sigma):
        """Tension fuera del plano, sigma_z, a partir de las tensiones en el plano.

        sigma puede tener cualquier forma mientras su ultimo eje sea
        [sigma_x, sigma_y, tau_xy]. Devuelve un arreglo con la forma de sigma
        sin el ultimo eje:

            deformacion plana (eps_z = 0): sigma_z = nu (sigma_x + sigma_y)
            tension plana:                 sigma_z = 0
        """
        sigma = numpy.asarray(sigma, dtype=float)

        # sigma[..., 0]: componente 0 del ultimo eje, cualquiera sea la forma del resto
        if self.model == "plane_strain":
            return self.nu * (sigma[..., 0] + sigma[..., 1])

        return numpy.zeros(sigma.shape[:-1])
