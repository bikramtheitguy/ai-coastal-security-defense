# Optional extra CA certificates for the image build

If the build machine sits behind a TLS-inspecting proxy (common on Government networks), place the proxy's
root CA certificate(s) here as `*.crt`. They are added to the image trust store for `npm ci` / `pip install`
and at runtime. `*.crt` files in this folder are git-ignored — never commit certificates or keys.
