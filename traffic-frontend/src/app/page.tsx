'use client'; // यह Next.js में Client Component बनाने के लिए ज़रूरी है

import { useState, useEffect } from 'react';

// Define the type for the vehicle counts data
interface VehicleCounts {
  car: number;
  motorcycle: number;
  bus: number;
  truck: number;
  total: number;
}

export default function Home() {
  const [videoStarted, setVideoStarted] = useState(false);
  const [counts, setCounts] = useState<VehicleCounts>({
    car: 0,
    motorcycle: 0,
    bus: 0,
    truck: 0,
    total: 0,
  });

  const backendUrl = 'http://127.0.0.1:5000'; // आपके Flask सर्वर का URL

  const startAnalysis = () => {
    setVideoStarted(true);
  };

  useEffect(() => {
    if (videoStarted) {
      // Fetch counts from the backend every 1 second
      const intervalId = setInterval(() => {
        fetch(`${backendUrl}/get_counts`)
          .then(response => response.json())
          .then(data => {
            setCounts(data);
          })
          .catch(error => console.error('Error fetching counts:', error));
      }, 1000);

      // Cleanup function to clear the interval when the component unmounts
      return () => clearInterval(intervalId);
    }
  }, [videoStarted]);

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>Real-Time Traffic Analysis</h1>
      <p>Click the button to start the video analysis stream from the backend.</p>

      {!videoStarted && (
        <button onClick={startAnalysis} style={{ padding: '10px 20px', fontSize: '16px' }}>
          Start Analysis
        </button>
      )}

      {videoStarted && (
        <div>
          <h2>Live Feed from Server</h2>
          <img
            src={`${backendUrl}/video_feed`}
            alt="Live Traffic Analysis"
            width="800"
            style={{ border: '2px solid #ccc' }}
          />

          <div style={{ marginTop: '20px' }}>
            <h3>Vehicle Counts</h3>
            <p><strong>Total:</strong> {counts.total}</p>
            <p><strong>Car:</strong> {counts.car}</p>
            <p><strong>Motorcycle:</strong> {counts.motorcycle}</p>
            <p><strong>Bus:</strong> {counts.bus}</p>
            <p><strong>Truck:</strong> {counts.truck}</p>
          </div>
        </div>
      )}
    </div>
  );
}