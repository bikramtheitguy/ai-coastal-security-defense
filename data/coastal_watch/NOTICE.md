# Public geography and synthetic exercise notices

Project code remains under the repository's MIT licence. Third-party geography has its own terms.

**Natural Earth:** public-domain 1:50 million coastline, repository version 4.1.0. Made with Natural Earth. [Terms](https://www.naturalearthdata.com/about/terms-of-use/). The registry records the pinned revision, upstream hash and clipped output hash. Generalised coastline is not a navigation chart or a precise coastal-distance baseline.

**State/UT geometry:** DataMeet India community / Election Commission of India, distributed by geoBoundaries under [CC BY 2.5 India](https://creativecommons.org/licenses/by/2.5/in/). Original coordinate arrays are retained; features are selected and descriptive properties added. The ADM1 metadata reports 2011 but includes later administrative units. Currency is unverified.

**District geometry:** geoBoundaries / Pathways Data Pvt. Ltd. / lgdirectory.gov.in, made available under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). The derived district database and added association fields are offered under that licence. The complete derived database is available in `india-reference.json`; its reproducible derivation is in `scripts/build_india_reference.py`. The pinned source is identified in `registry.json`.

District metadata represents 2021 and advertises 736 features; the pinned simplified source contains 735. The selected reference includes 229 districts associated with coastal states by polygon interior-point containment. It includes inland districts within those states. This is not an official current parent-code crosswalk. The exclusion ledger preserves unmatched source identifiers; most represent non-coastal states. Lakshadweep has no reliable district association in this mixed-vintage selection. No district is fabricated to fill the gap. Coordinates and polygon topology are unchanged.

**Exercise:** 44 fictional vessels, receipt faults and a fictional area, generated under the project's MIT licence. A vessel's exercise state is a scenario grouping, not its registration, jurisdiction or real deployment. Speed is in knots; positions use longitude/latitude decimal degrees, OGC:CRS84. All operational values are synthetic.

At rendering time only, copies of polygon rings are reordered for [D3's spherical winding convention](https://d3js.org/d3-geo). This does not change coordinate values, polygon interiors or the committed source-derived database.

These files contain no client maps, supplied Drive datasets, actual AIS feed or real security incidents. Names of geographic areas are context only. Do not use this POC for navigation or operational decisions.
