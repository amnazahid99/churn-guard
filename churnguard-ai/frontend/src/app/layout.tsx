import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ChurnGuard AI",
  description: "Full-stack churn prediction dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0a0a0f] text-white antialiased">{children}</body>
    </html>
  );
}
