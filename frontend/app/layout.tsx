import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Knowledge assistant",
  description: "Ask questions about one PDF, with the passages the answer came from.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="h-full antialiased">{children}</body>
    </html>
  );
}
