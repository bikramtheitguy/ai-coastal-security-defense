/** Static export: the FastAPI backend serves the built files, so the whole POC runs as one process/container.
 *  For `npm run dev`, API calls are proxied to the backend via NEXT_PUBLIC_API_BASE (default http://localhost:8000). */
const nextConfig = {
  output: process.env.NEXT_DEV_SERVER ? undefined : "export",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
  poweredByHeader: false,
};
export default nextConfig;
