"""Parser for the PSML pseudopotential file (https://esl.cecam.org/PSML) written by ONCVPSP.

The document gives the radial functions on a global grid (`<grid>`, unless a `<radfunc>` has its own):
the semilocal potentials, the valence and model core charges, the local potential, the nonlocal
projectors with their energies `ekb`, and the pseudo-wave-functions, each in a set
(`scalar_relativistic`, `lj` for j-channels, `spin_orbit`, ...). ONCVPSP puts the echo of its input in
`<provenance><input-file>`.

The projectors and their energies are not unique: any change of basis among the projectors of one set
and channel that keeps the nonlocal kernel `sum_ij |beta_i> ekb_ij <beta_j|` gives the same
pseudopotential.
"""

from typing import Any
from xml.etree import ElementTree

import numpy as np

__all__ = ("parse_psml",)

L_LETTERS = "spdfghi"


def _tag(element: ElementTree.Element) -> str:
    """Tag of `element` without its namespace."""
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ElementTree.Element, tag: str) -> list[ElementTree.Element]:
    return [child for child in element if _tag(child) == tag]


def _child(element: ElementTree.Element, tag: str) -> ElementTree.Element | None:
    children = _children(element, tag)
    return children[0] if children else None


def _numbers(element: ElementTree.Element | None) -> np.ndarray:
    if element is None or element.text is None:
        return np.array([])
    return np.array([float(x) for x in element.text.split()])


def _value(text: str) -> Any:
    """An attribute as an int or float if it is one, else as given."""
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _attributes(element: ElementTree.Element) -> dict[str, Any]:
    return {key: _value(value) for key, value in element.attrib.items()}


def _channel(attributes: dict[str, Any]) -> dict[str, Any]:
    """`l` as an integer and `j` (None if not given) of a PSML element's attributes."""
    ell = attributes.get("l")
    if isinstance(ell, str):
        ell = L_LETTERS.index(ell)
    j = attributes.get("j")
    return {"l": ell, "j": None if j is None else float(j)}


def _radfunc(element: ElementTree.Element) -> dict[str, Any]:
    """Data of the `<radfunc>` of `element`, and its own grid (None if it uses the global one)."""
    radfunc = _child(element, "radfunc")
    if radfunc is None:
        return {"r": None, "values": np.array([])}
    grid = _child(radfunc, "grid")
    return {
        "r": None if grid is None else _numbers(_child(grid, "grid-data")),
        "values": _numbers(_child(radfunc, "data")),
    }


def _functions(
    parent: ElementTree.Element | None, set_tag: str, item_tag: str
) -> list[dict]:
    """The radial functions `item_tag` of every set `set_tag` of `parent`."""
    if parent is None:
        return []
    functions = []
    for group in _children(parent, set_tag):
        set_name = group.attrib.get("set")
        for item in _children(group, item_tag):
            attributes = _attributes(item)
            functions.append(
                {
                    "set": set_name,
                    **attributes,
                    **_channel(attributes),
                    **_radfunc(item),
                }
            )
    return functions


def parse_psml(text: str) -> dict[str, Any]:
    """Parse the text of a PSML document.

    Args:
        text: Text of the document, from its XML declaration (or `<psml>`) on.

    Raises:
        ValueError: If the document is not PSML.

    Returns:
        dict:
            attributes (dict): Attributes of `<psml>` (version, units).
            provenance (list[dict]): Attributes of each `<provenance>` (creator, date, ...).
            input (str | None): Text of the `<input-file>` of the first provenance.
            header (dict): Attributes of `<header>` (atomic-label, atomic-number, z-pseudo, ...).
            exchange_correlation (dict):
                annotation (dict): Attributes of its `<annotation>`.
                functionals (list[dict]): Attributes of each libxc `<functional>`.
            valence_configuration (dict):
                attributes (dict): Attributes (total-valence-charge).
                shells (list[dict]): Attributes of each `<shell>`, with `l` as an integer.
            grid (np.ndarray): The global radial grid (bohr).
            grid_annotation (dict): Attributes of the grid's `<annotation>`.
            semilocal_potentials (list[dict]): Each `<slps>`: set, its attributes, `l` (integer), `j`
                (float or None), `r` (its own grid, or None) and `values`.
            local_potential (dict | None): Attributes of `<local-potential>`, `r` and `values`.
            projectors (list[dict]): Each `<proj>`, as the semilocal potentials, with `seq` and `ekb`.
            pseudo_wave_functions (list[dict]): Each `<pswf>`, as the semilocal potentials.
            valence_charge (dict | None): Attributes of `<valence-charge>`, `r` and `values`.
            core_charge (dict | None): Attributes of `<pseudocore-charge>`, `r` and `values`.
    """
    root = ElementTree.fromstring(text.strip())
    if _tag(root) != "psml":
        raise ValueError(f"not a PSML document: root element <{_tag(root)}>")
    provenance = _children(root, "provenance")
    input_file = _child(provenance[0], "input-file") if provenance else None
    header = _child(root, "header")
    xc = _child(header, "exchange-correlation") if header is not None else None
    annotation = _child(xc, "annotation") if xc is not None else None
    libxc = _child(xc, "libxc-info") if xc is not None else None
    valence = _child(header, "valence-configuration") if header is not None else None
    grid = _child(root, "grid")
    grid_annotation = _child(grid, "annotation") if grid is not None else None
    operator = _child(root, "pseudopotential-operator")
    local = _child(operator, "local-potential") if operator is not None else None

    def charge(tag: str) -> dict[str, Any] | None:
        element = _child(root, tag)
        if element is None:
            return None
        return {**_attributes(element), **_radfunc(element)}

    return {
        "attributes": _attributes(root),
        "provenance": [_attributes(p) for p in provenance],
        "input": None if input_file is None else input_file.text,
        "header": {} if header is None else _attributes(header),
        "exchange_correlation": {
            "annotation": {} if annotation is None else _attributes(annotation),
            "functionals": []
            if libxc is None
            else [_attributes(f) for f in _children(libxc, "functional")],
        },
        "valence_configuration": {
            "attributes": {} if valence is None else _attributes(valence),
            "shells": []
            if valence is None
            else [
                {**_attributes(shell), **_channel(_attributes(shell))}
                for shell in _children(valence, "shell")
            ],
        },
        "grid": _numbers(_child(grid, "grid-data") if grid is not None else None),
        "grid_annotation": {}
        if grid_annotation is None
        else _attributes(grid_annotation),
        "semilocal_potentials": _functions(root, "semilocal-potentials", "slps"),
        "local_potential": None
        if local is None
        else {**_attributes(local), **_radfunc(local)},
        "projectors": _functions(operator, "projectors", "proj"),
        "pseudo_wave_functions": _functions(root, "pseudo-wave-functions", "pswf"),
        "valence_charge": charge("valence-charge"),
        "core_charge": charge("pseudocore-charge"),
    }
