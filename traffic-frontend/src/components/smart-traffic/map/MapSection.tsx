'use client';

import { CrossPath, Theme } from '@/types/smart-traffic';
import InteractiveMap from './InteractiveMap';
import CrossPathDetails from './CrossPathDetails';

interface MapSectionProps {
  crossPaths: CrossPath[];
  selectedCrossPath: CrossPath | null;
  showPathDetails: boolean;
  theme: Theme;
  onCrossPathSelect: (path: CrossPath) => void;
  onShowPathDetails: (show: boolean) => void;
  backgroundImage?: string;
  showOverlay?: boolean;
  overlayOpacity?: number;
}

export default function MapSection({
  crossPaths,
  selectedCrossPath,
  showPathDetails,
  theme,
  onCrossPathSelect,
  onShowPathDetails,
  backgroundImage,
  showOverlay,
  overlayOpacity
}: MapSectionProps) {
  const handleCrossPathSelect = (path: CrossPath) => {
    onCrossPathSelect(path);
    onShowPathDetails(true);
  };

  const handleClose = () => {
    onShowPathDetails(false);
  };

  return (
    <div>
      <h2 style={{ color: theme.primary, marginBottom: '20px' }}>Bhubaneswar City Traffic Map</h2>
      
      <div style={{ display: 'flex', gap: '30px' }}>
        <InteractiveMap
          crossPaths={crossPaths}
          theme={theme}
          onCrossPathSelect={handleCrossPathSelect}
          backgroundImage={backgroundImage}
          showOverlay={showOverlay}
          overlayOpacity={overlayOpacity}
        />

        {showPathDetails && selectedCrossPath && (
          <CrossPathDetails
            crossPath={selectedCrossPath}
            theme={theme}
            onClose={handleClose}
          />
        )}
      </div>
    </div>
  );
}