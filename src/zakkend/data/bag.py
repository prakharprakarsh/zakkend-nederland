"""BAG (Basisregistratie Adressen en Gebouwen) loader — Phase 2 stub.

PDOK offers BAG as a WFS service:
    https://service.pdok.nl/lv/bag/wfs/v2_0

For Phase 1 we ship a stub that documents the exact request contract we'll
hit in Phase 2, so the rest of the pipeline can already call through a
consistent interface.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BAGBuilding:
    """Subset of BAG attributes we care about for subsidence modelling."""

    identificatie: str  # BAG unique id
    postcode: str
    huisnummer: int
    lat: float
    lon: float
    bouwjaar: int  # construction year
    oppervlakte: float  # floor area (m²)
    gebruiksdoel: str  # e.g. "woonfunctie"


def lookup_by_postcode_huisnummer(
    postcode: str, huisnummer: int
) -> BAGBuilding | None:
    """Phase 2: query PDOK WFS for a specific address.

    Raises
    ------
    NotImplementedError
        Phase 1 does not hit PDOK yet — synthetic data is used instead.
    """
    raise NotImplementedError(
        "Phase 2: will call PDOK BAG WFS. "
        "See https://www.pdok.nl/geo-services/-/article/basisregistratie-adressen-en-gebouwen-ba-"
    )
