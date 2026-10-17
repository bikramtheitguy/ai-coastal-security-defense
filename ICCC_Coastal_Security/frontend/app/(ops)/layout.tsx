"use client";
import { Suspense } from "react";
import { Shell } from "@/components/Shell";

export default function OpsLayout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<div className="empty">Loading…</div>}>
      <Shell>{children}</Shell>
    </Suspense>
  );
}
