import type { Metadata } from 'next';
import './globals.css'; 

export const metadata: Metadata = {
  title: 'SIH Traffic Management System',
  description: 'Advanced traffic management system with real-time monitoring and AI-powered analytics',
  keywords: 'traffic management, smart city, real-time monitoring, AI analytics, traffic control',
  authors: [{ name: 'SIH Traffic Management Team' }],
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
    },
  },
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
        <meta name="description" content="Advanced traffic management system with real-time monitoring and AI-powered analytics" />
        <link rel="icon" href="/favicon.ico" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
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