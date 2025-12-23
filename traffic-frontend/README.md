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

Create a `Dockerfile` for containerized deployment:

```Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
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