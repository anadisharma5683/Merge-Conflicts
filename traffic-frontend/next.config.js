/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable production optimizations
  productionBrowserSourceMaps: false,
  // Optimize images
  images: {
    unoptimized: true, // For deployment without image optimization if needed
  },
  // Enable compression
  compress: true,
  // Optimize webpack
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      };
    }
    return config;
  },
  // Optimize output
  output: 'standalone', // For standalone deployment
};

module.exports = nextConfig;