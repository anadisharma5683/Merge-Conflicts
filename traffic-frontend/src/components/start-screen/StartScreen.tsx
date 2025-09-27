'use client';

interface StartScreenProps {
  onStartAnalysis: () => void;
}

export default function StartScreen({ onStartAnalysis }: StartScreenProps) {
  return (
    <div style={{ textAlign: 'center', padding: '40px' }}>
      <p style={{ fontSize: '1.2em', marginBottom: '30px' }}>
        Click the button to start the video analysis stream from the backend.
      </p>
      <button 
        onClick={onStartAnalysis}
        style={{
          background: 'linear-gradient(45deg, #4caf50, #8bc34a)',
          border: 'none',
          borderRadius: '25px',
          padding: '15px 30px',
          fontSize: '18px',
          color: 'white',
          fontWeight: 'bold',
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)'
        }}
        onMouseOver={(e) => {
          e.currentTarget.style.transform = 'translateY(-2px)';
          e.currentTarget.style.boxShadow = '0 6px 20px rgba(0, 0, 0, 0.3)';
        }}
        onMouseOut={(e) => {
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.2)';
        }}
      >
        🚀 Start Analysis
      </button>
    </div>
  );
}