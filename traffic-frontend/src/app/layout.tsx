import type { Metadata } from 'next';
import './globals.css'; // अगर आपके पास यह फ़ाइल है तो इसे रखें

export const metadata: Metadata = {
  title: 'Traffic Analysis App',
  description: 'A web application for real-time traffic analysis.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}