from source.reference_elements.tria import Tria
from source.reference_elements.quad import Quad
from source.reference_elements.line import Line


# Nombre del elemento -> (clase, orden de interpolacion)
ELEMENTS = {
    "CST": (Tria, 1),
    "LST": (Tria, 2),
    "Q4": (Quad, 1),
    "Q8": (Quad, 2),
    "L2": (Line, 1),
    "L3": (Line, 2),
}

# Nombres alternativos, para que sigan sirviendo las mallas que usan T3 y T6
ALIASES = {"T3": "CST", "T6": "LST"}


def make_reference_element(name):
    """Devuelve el elemento de referencia que corresponde al nombre, por
    ejemplo make_reference_element("Q8") es Quad("Q8").
    """
    if name in ALIASES:
        name = ALIASES[name]

    if name not in ELEMENTS:
        raise ValueError("Unknown reference element: " + str(name))

    element_class = ELEMENTS[name][0]

    return element_class(name)
