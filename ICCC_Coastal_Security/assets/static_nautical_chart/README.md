# Static nautical chart — offline fallback

**LOCAL STATIC NAUTICAL REFERENCE — NOT FOR NAVIGATION**

The build specification refers to a supplied Odisha static nautical chart. **No chart file was supplied with this
repository**, so a clearly labelled placeholder (`placeholder_no_chart_supplied.svg`) is shown instead.

To use an authorised chart image:

1. Place a `.png`, `.jpg`, `.webp` or `.svg` file in this folder (any name not starting with `placeholder`).
2. Restart the backend. It serves the first image it finds at `/api/public/static-chart`; the
   *COP › Static Nautical Chart* view displays it with zoom controls.

The image is deliberately **not georeferenced** and operational layers are **not** overlaid on it, because no verified
calibration is available. Georeferencing (GCPs + projection) should only be added once the chart source and its calibration
have been verified by the chart owner. Confirm usage rights before adding any chart (for example Indian National
Hydrographic Office products are subject to licensing).
