# SIH Traffic Management System

Advanced traffic management system with real-time monitoring and AI-powered analytics for smart city initiatives.

## Features

- Real-time traffic monitoring
- Interactive map with congestion visualization
- Live video feed analysis
- Traffic signal control
- Warning and alert system
- Accident reporting and management

## Technology Stack

- Next.js 14 with App Router
- React 18
- TypeScript
- Tailwind CSS
- Recharts for data visualization
- Lucide React for icons

## Deployment

### Prerequisites

- Node.js 18 or higher
- npm 8 or higher

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

### Production Build

```bash
npm run build
```

### Production Deployment

```bash
npm run start
```

## Environment Variables

Create a `.env.local` file in the root directory:

```
NEXT_PUBLIC_API_URL=https://your-api-url.com
NEXT_PUBLIC_MAP_API_KEY=your-map-api-key
```

## Docker Deployment (Optional)

The project includes a `Dockerfile` for containerized deployment:

```Dockerfile
# Use official Node.js runtime as a parent image
FROM node:18-alpine

# Set environment variables
ENV NODE_ENV=production

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy the rest of the application code
COPY . .

# Build the application
RUN npm run build

# Expose the port the app runs on
EXPOSE 3000

# Define the command to run the application
CMD ["node", ".next/server.js"]
```

## Production Optimizations

- Server-side rendering for better SEO
- Image optimization
- Code splitting
- Bundle optimization
- Static site generation capability

## API Integration

The application is designed to connect to a backend API for:
- Traffic data
- Video stream processing
- Signal control
- User authentication

## License

MIT License