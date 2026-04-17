"""Dutch foundation remediation knowledge base.

Curated from KCAF (Kenniscentrum Aanpak Funderingsproblematiek),
municipal funding schemes, and published guidance.

This module serves as the retrieval layer for the LangGraph agent —
structured knowledge that the report-drafting node draws from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RemediationOption:
    name: str
    description: str
    applicable_when: str
    estimated_cost_eur: str
    typical_duration: str
    urgency: str  # "immediate", "within_1_year", "monitoring"


@dataclass(frozen=True)
class FundingScheme:
    name: str
    provider: str
    description: str
    url: str
    max_amount_eur: str
    eligibility: str


@dataclass(frozen=True)
class InspectionBody:
    name: str
    description: str
    url: str
    service_type: str


# ──────────────────── Remediation options ────────────────────

REMEDIATION_OPTIONS: list[RemediationOption] = [
    RemediationOption(
        name="Funderingsherstel (pile replacement)",
        description=(
            "Complete foundation replacement: install new steel or concrete "
            "micro-piles through the existing structure down to the sand layer. "
            "The building is temporarily supported on hydraulic jacks."
        ),
        applicable_when="Wooden pile rot confirmed; foundation bearing capacity below safe threshold",
        estimated_cost_eur="€40,000–€120,000 per home (depends on size and access)",
        typical_duration="4–8 weeks per home",
        urgency="immediate",
    ),
    RemediationOption(
        name="Grondwaterpeilbeheer (groundwater management)",
        description=(
            "Install or adjust local groundwater management to keep water levels "
            "above wooden pile heads. Prevents further rot by keeping piles submerged. "
            "Often a municipal-level intervention."
        ),
        applicable_when="Wooden pile foundation with dropping groundwater table",
        estimated_cost_eur="€5,000–€15,000 per home (shared infrastructure reduces cost)",
        typical_duration="3–6 months (planning + installation)",
        urgency="within_1_year",
    ),
    RemediationOption(
        name="Bodeminjectie (soil injection / ground stabilization)",
        description=(
            "Inject resin or grout into the soil beneath the foundation to "
            "stabilize and lift settled areas. Less invasive than pile replacement."
        ),
        applicable_when="Moderate settlement on clay/sandy soils; strip or slab foundations",
        estimated_cost_eur="€15,000–€40,000",
        typical_duration="1–3 weeks",
        urgency="within_1_year",
    ),
    RemediationOption(
        name="Monitoring programme",
        description=(
            "Install tilt sensors, crack monitors, and groundwater level loggers. "
            "Track deformation over 12–24 months to determine if intervention is needed."
        ),
        applicable_when="Risk is moderate; no visible damage yet; preventive assessment",
        estimated_cost_eur="€2,000–€5,000 for installation + annual monitoring",
        typical_duration="Ongoing (12–24 month minimum)",
        urgency="monitoring",
    ),
    RemediationOption(
        name="Drainage improvement",
        description=(
            "Improve surface and subsurface drainage to reduce seasonal "
            "groundwater fluctuations that accelerate pile rot and peat oxidation."
        ),
        applicable_when="High groundwater variability; seasonal flooding risk",
        estimated_cost_eur="€3,000–€10,000",
        typical_duration="2–4 weeks",
        urgency="within_1_year",
    ),
]


# ──────────────────── Funding schemes ────────────────────

FUNDING_SCHEMES: list[FundingScheme] = [
    FundingScheme(
        name="Nationaal Fonds Funderingsherstel (NFF)",
        provider="Rijksoverheid (national government)",
        description=(
            "Low-interest loans (1.5–2.5%) for foundation repair. Available to "
            "homeowners who cannot fully finance repair through regular channels."
        ),
        url="https://www.funderingsherstelfonds.nl/",
        max_amount_eur="Up to €120,000 per home",
        eligibility="Owner-occupied homes with confirmed foundation problems",
    ),
    FundingScheme(
        name="KCAF Funderingsloket",
        provider="Kenniscentrum Aanpak Funderingsproblematiek",
        description=(
            "Free advice and guidance for homeowners facing foundation problems. "
            "Helps navigate inspection, repair options, and available funding."
        ),
        url="https://www.kcaf.nl/funderingsloket/",
        max_amount_eur="Free advisory service (no direct funding)",
        eligibility="All homeowners in the Netherlands",
    ),
    FundingScheme(
        name="Gemeentelijke subsidie funderingsherstel",
        provider="Municipal governments (varies by municipality)",
        description=(
            "Several municipalities offer additional subsidies or project coordination "
            "for neighborhood-level foundation repair. Gouda, Rotterdam, Zaanstad, "
            "Dordrecht, and Schiedam have active programmes."
        ),
        url="https://www.kcaf.nl/gemeenten/",
        max_amount_eur="€5,000–€30,000 (varies by municipality)",
        eligibility="Homeowners in participating municipalities",
    ),
    FundingScheme(
        name="Energiebespaarlening (combined renovation)",
        provider="Nationaal Warmtefonds",
        description=(
            "When foundation repair is combined with energy-efficiency improvements, "
            "homeowners may qualify for favorable loans from the Nationaal Warmtefonds."
        ),
        url="https://www.warmtefonds.nl/",
        max_amount_eur="Up to €65,000 for combined renovation",
        eligibility="Owner-occupied homes undergoing energy renovation",
    ),
]


# ──────────────────── Inspection bodies ────────────────────

INSPECTION_BODIES: list[InspectionBody] = [
    InspectionBody(
        name="KCAF-certified inspection firms",
        description="Foundation inspection firms certified by KCAF following the F3O protocol",
        url="https://www.kcaf.nl/funderingsonderzoek/",
        service_type="Foundation inspection (F3O protocol)",
    ),
    InspectionBody(
        name="SHR (Stichting Hout Research)",
        description="Specialized in wooden pile condition assessment via core sampling",
        url="https://www.shr.nl/",
        service_type="Wood quality / pile rot assessment",
    ),
    InspectionBody(
        name="Fugro",
        description="Geotechnical investigations including soil sampling and pile testing",
        url="https://www.fugro.com/",
        service_type="Geotechnical / soil investigation",
    ),
    InspectionBody(
        name="Wareco",
        description="Groundwater monitoring and management solutions",
        url="https://www.wareco.nl/",
        service_type="Groundwater monitoring",
    ),
]


# ──────────────────── Lookup functions ────────────────────

def get_applicable_remediations(
    risk_class: str,
    foundation_type: str,
    soil_type: str,
) -> list[RemediationOption]:
    """Return remediation options relevant to this building's risk profile."""
    results = []

    if risk_class == "critical":
        # Critical: full pile replacement + groundwater management
        results.extend([
            r for r in REMEDIATION_OPTIONS
            if r.urgency == "immediate"
        ])
        if foundation_type == "wooden_pile":
            results.extend([
                r for r in REMEDIATION_OPTIONS
                if "grondwater" in r.name.lower() or "groundwater" in r.description.lower()
            ])

    elif risk_class == "high":
        # High: soil injection, groundwater management, monitoring
        results.extend([
            r for r in REMEDIATION_OPTIONS
            if r.urgency in ("within_1_year", "monitoring")
        ])

    elif risk_class == "moderate":
        # Moderate: monitoring + drainage
        results.extend([
            r for r in REMEDIATION_OPTIONS
            if r.urgency == "monitoring" or "drainage" in r.name.lower()
        ])

    else:
        # Low: monitoring only
        results.extend([
            r for r in REMEDIATION_OPTIONS
            if r.urgency == "monitoring"
        ])

    # Deduplicate by name
    seen = set()
    unique = []
    for r in results:
        if r.name not in seen:
            seen.add(r.name)
            unique.append(r)

    return unique


def get_applicable_funding(risk_class: str) -> list[FundingScheme]:
    """Return funding schemes relevant to the risk level."""
    if risk_class in ("critical", "high"):
        return FUNDING_SCHEMES  # All schemes potentially applicable
    elif risk_class == "moderate":
        return [f for f in FUNDING_SCHEMES if "loket" in f.name.lower() or "gemeente" in f.name.lower()]
    else:
        return [FUNDING_SCHEMES[1]]  # KCAF Funderingsloket (free advice)


def get_inspection_bodies(foundation_type: str) -> list[InspectionBody]:
    """Return relevant inspection bodies based on foundation type."""
    bodies = [INSPECTION_BODIES[0]]  # KCAF-certified always included
    if foundation_type == "wooden_pile":
        bodies.append(INSPECTION_BODIES[1])  # SHR for wood assessment
    bodies.append(INSPECTION_BODIES[2])  # Fugro for geotechnical
    bodies.append(INSPECTION_BODIES[3])  # Wareco for groundwater
    return bodies
