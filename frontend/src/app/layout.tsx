import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Activity } from "lucide-react";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ImmunoPredict | Clinical Dashboard",
  description: "AI-Mechanistic Vaccine Response Prediction",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-background text-foreground min-h-screen flex flex-col`}>
        {/* Sleek Medical Navigation Bar */}
        <header className="bg-card border-border border-b sticky top-0 z-50 shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="bg-[var(--primary)] p-2 rounded-lg">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold tracking-tight text-foreground">
                Immuno<span className="text-primary">Predict</span>
              </span>
            </div>
            
            <nav className="flex space-x-6 text-sm font-medium text-muted-foreground">
              <a href="/" className="text-primary font-semibold hover:text-primary/80 transition-colors">Prediction Suite</a>
              <a href="/history" className="hover:text-primary transition-colors">Audit Log</a>
            </nav>
          </div>
        </header>

        {/* Main Application Area */}
        <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
          {children}
        </main>
      </body>
    </html>
  );
}
