# Data sources

All data sources used or planned are **open and Dutch-official**.

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
- Use: model calibration + skill verification

## Refresh cadence

| Source   | Typical update cadence |
| -------- | ---------------------- |
| BAG      | Daily                  |
| TNO soil | Multi-year static      |
| BDK 2.0  | Annual (Sentinel pass) |
| KNMI     | Daily                  |
