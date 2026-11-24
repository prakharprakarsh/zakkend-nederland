# Data sources

All data sources used or planned are open Dutch government data.

## Feature provenance

What is measured vs. simulated in the current model:

| Feature | Source |
|---------|--------|
| `year_built` | Measured — PDOK BAG `bouwjaar` |
| `building_age` | Derived — `REFERENCE_YEAR − year_built` |
| `soil_type` | Hybrid — BRO Bodemkaart spatial join for 1,031 of 3,828 buildings (26.9%); coordinate rule for the rest (urban areas aren't covered by the 1:50,000 map) |
| `peat_thickness_m` | Simulated — seeded by coordinates, TNO thickness ranges as priors |
| `groundwater_depth_m` | Simulated — per-soil-type normal distributions |
| `groundwater_variability` | Simulated — per-soil-type beta distributions |
| `insar_deformation_mm_yr` | Simulated — seeded by coordinates, means from BodemDalingsKaart regional rates |
| `drought_exposure_index` | Simulated — latitude gradient from KNMI maps |
| `distance_to_water_m` | Simulated — exponential draw, east/west split |
| `neighborhood_damage_rate` | Simulated — beta draw per spatial hash bucket |
| `foundation_type` | Simulated — weighted by construction era and soil type |

Building centroids (lat/lon) are measured from BAG and drive the coordinate-based estimators, but are not model features themselves.

## Phase 2: Buildings & addresses

**PDOK BAG (Basisregistratie Adressen en Gebouwen)**

- URL: https://www.pdok.nl/geo-services/-/article/basisregistratie-adressen-en-gebouwen-ba-
- Access: WFS / WMS / downloadable extract
- Licence: CC0 — free re-use including commercial
- Fields used: `identificatie`, `postcode`, `huisnummer`, `bouwjaar`, `oppervlakte`, geometry

## Phase 2: Soil composition

**TNO DINOloket — Bodemkaart 1:50.000**

- URL: https://www.dinoloket.nl/
- Access: WCS + bulk download
- Licence: CC BY 4.0
- Fields used: primary soil class, peat thickness, groundwater regime

## Phase 2: Ground deformation

**BodemDalingsKaart 2.0 (Sentinel-1 InSAR, processed)**

- URL: https://bodemdalingskaart.nl/
- Access: WMS / downloadable
- Licence: CC BY 4.0
- Fields used: vertical deformation (mm/yr), time-series since 2015

## Phase 3: Climate

**KNMI (Royal Netherlands Meteorological Institute)**

- URL: https://dataplatform.knmi.nl/
- Access: REST API with free token
- Licence: CC BY 4.0
- Fields used: daily precipitation, reference evapotranspiration, KNMI drought index

## Phase 5: Ground truth

**KCAF known-damage dataset**

- Source: Kenniscentrum Aanpak Funderingsproblematiek, via research partnerships
- Access: On request for research; aggregated municipal data is public
- Use: real-world skill verification (comparison against known damage outcomes)

## Refresh cadence

| Source   | Typical update cadence |
| -------- | ---------------------- |
| BAG      | Daily                  |
| TNO soil | Multi-year static      |
| BDK 2.0  | Annual (Sentinel pass) |
| KNMI     | Daily                  |
