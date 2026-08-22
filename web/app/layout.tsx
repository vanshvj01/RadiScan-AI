import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'RadiScan AI',
  description:
    'Uncertainty-aware, explainable chest radiograph analysis. Decision support only.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <div className="mx-auto max-w-6xl px-4 py-6">{children}</div>
      </body>
    </html>
  );
}
