import type { Metadata } from 'next';
import './globals.css'; 

export const metadata: Metadata = {
  title: 'Traffic Analysis App - Real-time Vehicle Detection',
  description: 'A web application for real-time traffic analysis with play/pause controls, counter reset, and performance optimization.',
  keywords: 'traffic analysis, vehicle detection, real-time, YOLO, computer vision',
  authors: [{ name: 'Traffic Analysis Team' }],
  viewport: 'width=device-width, initial-scale=1',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body 
        style={{
          margin: 0,
          padding: 0,
          fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
          backgroundColor: '#f5f5f5'
        }}
      >
        {children}
      </body>
    </html>
  );
}