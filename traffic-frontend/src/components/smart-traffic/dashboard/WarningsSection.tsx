import React from 'react';
import { Theme } from '@/types/smart-traffic';

interface WarningsSectionProps {
  theme: Theme;
}

const WarningsSection: React.FC<WarningsSectionProps> = ({ theme }) => {
  return (
    <div style={{
      background: theme.background,
      borderRadius: '15px',
      padding: '50px',
      textAlign: 'center',
      border: `2px solid ${theme.primary}10`
    }}>
      <h2 style={{ color: theme.primary, marginBottom: '20px', fontSize: '24px', fontWeight: 'bold' }}>
        Traffic Warnings Section
      </h2>
      <p style={{ color: theme.neutral, fontSize: '16px' }}>
        This section is being implemented. The modular structure allows easy addition of new components.
      </p>
    </div>
  );
};

export default WarningsSection;