// src/components/Navbar.tsx
"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";

export function Navbar() {
  return (
    <nav className="w-full flex items-center justify-between px-6 py-3 shadow-md bg-white">
      <Link href="/" className="text-xl font-bold">
        Traffic Dashboard
      </Link>
      <div className="flex gap-3">
        <Link href="/stats">
          <Button variant="outline">Statistics</Button>
        </Link>
        <Link href="/">
          <Button>Live Feed</Button>
        </Link>
      </div>
    </nav>
  );
}
