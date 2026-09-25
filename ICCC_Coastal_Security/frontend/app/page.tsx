"use client";
import { useEffect } from "react";
import { useAuth } from "@/lib/auth";

export default function Home() {
  const { user, loading } = useAuth();
  useEffect(() => {
    if (loading) return;
    location.replace(user ? user.landing : "/login/");
  }, [user, loading]);
  return <div className="empty">Loading…</div>;
}
