import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CodeQ-Mate",
  description: "Context-aware question answering for software repositories",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
