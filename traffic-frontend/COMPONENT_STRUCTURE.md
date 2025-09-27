# Traffic Frontend - Component Structure

This document outlines the modular component structure of the traffic frontend application.

## 📁 Folder Structure

```
src/
├── app/
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx                    # Main entry point (now simplified)
├── components/
│   ├── ui/                         # Existing UI components
│   │   ├── Navbar.tsx
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   └── input.tsx
│   ├── dashboard/                  # Main dashboard container
│   │   ├── Dashboard.tsx
│   │   └── index.ts
│   ├── start-screen/              # Initial start screen
│   │   ├── StartScreen.tsx
│   │   └── index.ts
│   ├── video-feed/                # Live video stream component
│   │   ├── VideoFeed.tsx
│   │   └── index.ts
│   ├── controls/                  # Playback and frame controls
│   │   ├── Controls.tsx
│   │   └── index.ts
│   ├── status-panel/             # Status information display
│   │   ├── StatusPanel.tsx
│   │   └── index.ts
│   └── vehicle-counts/           # Vehicle statistics display
│       ├── VehicleCountsDisplay.tsx
│       └── index.ts
├── hooks/                         # Custom React hooks
│   ├── useTrafficData.ts         # Main traffic data management hook
│   └── index.ts
├── types/                         # TypeScript type definitions
│   ├── traffic.ts                # Traffic-related types
│   └── index.ts
└── lib/
    └── utils.ts                  # Existing utility functions
```

## 🧩 Component Overview

### 1. **Dashboard** (`src/components/dashboard/`)
- **Purpose**: Main container component that orchestrates all other components
- **Responsibilities**: 
  - Manages global state using the `useTrafficData` hook
  - Handles the main layout and component composition
  - Controls the flow between start screen and analysis view

### 2. **StartScreen** (`src/components/start-screen/`)
- **Purpose**: Initial screen displayed before starting video analysis
- **Props**: 
  - `onStartAnalysis`: Callback function to start the analysis

### 3. **VideoFeed** (`src/components/video-feed/`)
- **Purpose**: Displays the live video stream from the backend
- **Props**:
  - `backendUrl`: The URL of the backend server for video streaming

### 4. **Controls** (`src/components/controls/`)
- **Purpose**: Provides playback controls and performance settings
- **Props**:
  - `isPlaying`: Current playback state
  - `isLoading`: Loading state for API calls
  - `frameSkip`: Current frame skip value
  - `currentFrame`: Current frame number
  - `onTogglePlayPause`: Play/pause toggle callback
  - `onResetCounters`: Reset counters callback
  - `onUpdateFrameSkip`: Frame skip update callback

### 5. **StatusPanel** (`src/components/status-panel/`)
- **Purpose**: Displays current status information (playback state, frame info)
- **Props**:
  - `isPlaying`: Current playback state
  - `currentFrame`: Current frame number
  - `frameSkip`: Current frame skip value

### 6. **VehicleCountsDisplay** (`src/components/vehicle-counts/`)
- **Purpose**: Shows vehicle count statistics in a grid layout
- **Props**:
  - `counts`: Object containing all vehicle counts

## 🪝 Custom Hooks

### `useTrafficData`
- **Location**: `src/hooks/useTrafficData.ts`
- **Purpose**: Manages all traffic-related state and API interactions
- **Parameters**: 
  - `videoStarted`: Boolean indicating if video analysis has started
- **Returns**: Object containing:
  - State variables (counts, isPlaying, frameSkip, etc.)
  - API functions (togglePlayPause, resetCounters, updateFrameSkip)
  - Backend URL and error handling

## 📋 Type Definitions

### Traffic Types (`src/types/traffic.ts`)
- `VehicleCounts`: Interface for vehicle count data
- `BackendStatus`: Interface for backend response structure
- `VehicleDisplayItem`: Interface for vehicle display configuration

## 🚀 Benefits of This Structure

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Reusability**: Components can be easily reused in different contexts
3. **Maintainability**: Changes to one section don't affect others
4. **Testability**: Each component can be tested in isolation
5. **Type Safety**: Centralized type definitions ensure consistency
6. **Custom Hooks**: Business logic is separated from UI components

## 🛠️ How to Edit Components

To edit specific sections of the application:

1. **Dashboard Layout**: Edit `src/components/dashboard/Dashboard.tsx`
2. **Video Display**: Edit `src/components/video-feed/VideoFeed.tsx`
3. **Controls Panel**: Edit `src/components/controls/Controls.tsx`
4. **Vehicle Statistics**: Edit `src/components/vehicle-counts/VehicleCountsDisplay.tsx`
5. **Status Information**: Edit `src/components/status-panel/StatusPanel.tsx`
6. **Start Screen**: Edit `src/components/start-screen/StartScreen.tsx`
7. **Business Logic**: Edit `src/hooks/useTrafficData.ts`
8. **Type Definitions**: Edit `src/types/traffic.ts`

## 📦 Import Examples

```typescript
// Import main dashboard
import Dashboard from '@/components/dashboard';

// Import individual components
import VideoFeed from '@/components/video-feed';
import Controls from '@/components/controls';
import VehicleCountsDisplay from '@/components/vehicle-counts';

// Import custom hook
import { useTrafficData } from '@/hooks';

// Import types
import { VehicleCounts, BackendStatus } from '@/types';
```

This modular structure makes the codebase much more maintainable and allows for easy editing of individual sections without affecting the entire application.