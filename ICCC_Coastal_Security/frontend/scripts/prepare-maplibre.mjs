import { copyFileSync, cpSync, existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

/** Next cannot infer MapLibre 6's dynamically computed module-worker URL.
 * Ship the installed, integrity-locked distribution alongside the static export.
 * Preserve relative worker/chunk imports and the upstream license, without a CDN.
 */
export function prepareMapLibre() {
  const packageRoot = fileURLToPath(new URL("../node_modules/maplibre-gl/", import.meta.url));
  const { version } = JSON.parse(readFileSync(join(packageRoot, "package.json"), "utf8"));
  if (!/^\d+\.\d+\.\d+$/.test(version)) throw new Error("Expected a pinned stable MapLibre version");
  const dist = join(packageRoot, "dist");
  if (!existsSync(join(dist, "maplibre-gl-worker.mjs"))) throw new Error("MapLibre worker missing from installed package");
  const output = fileURLToPath(new URL(`../public/vendor/maplibre-${version}/`, import.meta.url));
  mkdirSync(output, { recursive: true });
  cpSync(dist, output, {
    recursive: true,
    filter: (source) => statSync(source).isDirectory() || /\.(m?js|wasm|json|css)$/.test(source),
  });
  copyFileSync(join(packageRoot, "LICENSE.txt"), join(output, "LICENSE.txt"));
}
