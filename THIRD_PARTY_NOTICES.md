# Third-party attribution

The repository's own code retains the existing MIT license and copyright notice in LICENSE.

The supported runtime uses Streamlit, pandas and Pillow plus their transitive dependencies. Optional research and development tools have their own licenses. Consult package distribution metadata and upstream notices; the root MIT license does not replace them.

Exact supported package versions and distribution hashes are recorded in uv.lock and the generated requirements exports. A release SBOM and comprehensive redistribution review are future work; this file is not a completed license clearance.

The historical committed venv/ includes third-party source, binaries and package notices. It is retained unchanged, is unsupported and must be excluded from release packages by an explicit allowlist.

Original CSV/chart/model generation provenance is incomplete; see docs/DATA_PROVENANCE.md. Future datasets, imagery, basemaps, pretrained weights and provider feeds require separate attribution and usage-rights review.

Browser verification uses Microsoft's Playwright (Apache-2.0) and pytest in a separate test environment. Their Python dependency versions and hashes are in `browser_tests/requirements.txt`. Playwright-managed Chromium carries its own notices; browser binaries are installed on the CI runner and are not redistributed in this repository. See [browser verification](docs/BROWSER_QA.md). This addition does not change the Streamlit runtime dependencies or complete the outstanding release SBOM/licence review.
