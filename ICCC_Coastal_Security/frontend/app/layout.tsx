import type { Metadata, Viewport } from "next";
import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import { SimBanner } from "@/components/Shell";

export const metadata: Metadata = {
  title: "Coastal Security ICCC — MDA Platform (POC)",
  description: "AI-Enabled Integrated Coastal Security & Maritime Domain Awareness Platform — Proof of Concept. SIMULATED / POC DATA.",
};
export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#0a1017" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <SimBanner />
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
