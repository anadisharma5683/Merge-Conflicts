'use client';

import { useSmartTrafficSystem } from '@/hooks/useSmartTrafficSystem';
import { smartTrafficTheme } from '@/lib/smart-traffic-theme';
import LoginScreen from './login/LoginScreen';
import Header from './layout/Header';
import Sidebar from './layout/Sidebar';
import MapSection from './map/MapSection';
import VideoAnalysisSection from './video-analysis/VideoAnalysisSection';
import SignalsControlSection from './signals/SignalsControlSection';
import { WarningsSection } from './dashboard';

export default function SmartTrafficSystem() {
  const {
    // Authentication
    isLoggedIn,
    setIsLoggedIn,
    username,
    setUsername,
    password,
    setPassword,
    loginError,
    handleLogin,
    
    // Navigation
    activeSection,
    setActiveSection,
    
    // Map
    selectedCrossPath,
    setSelectedCrossPath,
    showPathDetails,
    setShowPathDetails,
    crossPaths,
    
    // Video
    isVideoPlaying,
    setIsVideoPlaying,
    videoVolume,
    setVideoVolume,
    isFullscreen,
    setIsFullscreen,
    
    // Traffic Signals
    trafficSignals,
    overrideMode,
    setOverrideMode,
    overrideLogs,
    handleSignalOverride,
    
    // Analytics
    trafficStats
  } = useSmartTrafficSystem();

  const theme = smartTrafficTheme;

  // Login Screen
  if (!isLoggedIn) {
    return (
      <LoginScreen
        username={username}
        password={password}
        loginError={loginError}
        theme={theme}
        onUsernameChange={setUsername}
        onPasswordChange={setPassword}
        onLogin={handleLogin}
      />
    );
  }

  // Main Dashboard
  return (
    <div style={{
      minHeight: '100vh',
      background: theme.accent,
      fontFamily: 'system-ui, -apple-system, sans-serif',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <Header theme={theme} onLogout={() => setIsLoggedIn(false)} />

      <div style={{ display: 'flex', flex: 1 }}>
        <Sidebar
          activeSection={activeSection}
          theme={theme}
          onSectionChange={setActiveSection}
        />

        <main style={{ flex: 1, padding: '30px', overflow: 'auto' }}>
          {activeSection === 'map' && (
            <MapSection
              crossPaths={crossPaths}
              selectedCrossPath={selectedCrossPath}
              showPathDetails={showPathDetails}
              theme={theme}
              onCrossPathSelect={setSelectedCrossPath}
              onShowPathDetails={setShowPathDetails}
            />
          )}

          {activeSection === 'video' && (
            <VideoAnalysisSection
              isVideoPlaying={isVideoPlaying}
              videoVolume={videoVolume}
              isFullscreen={isFullscreen}
              trafficStats={trafficStats}
              theme={theme}
              onPlayPause={() => setIsVideoPlaying(!isVideoPlaying)}
              onVolumeChange={setVideoVolume}
              onFullscreen={() => setIsFullscreen(!isFullscreen)}
            />
          )}

          {activeSection === 'signals' && (
            <SignalsControlSection
              trafficSignals={trafficSignals}
              overrideMode={overrideMode}
              overrideLogs={overrideLogs}
              theme={theme}
              onOverrideModeToggle={() => setOverrideMode(!overrideMode)}
              onSignalOverride={handleSignalOverride}
            />
          )}

          {activeSection === 'warnings' && (
            <WarningsSection />
          )}

          {/* Placeholder for other sections */}
          {!['map', 'video', 'signals', 'warnings'].includes(activeSection) && (
            <div style={{
              background: theme.background,
              borderRadius: '15px',
              padding: '50px',
              textAlign: 'center',
              border: `2px solid ${theme.primary}10`
            }}>
              <h2 style={{ color: theme.primary, marginBottom: '20px' }}>
                {activeSection.charAt(0).toUpperCase() + activeSection.slice(1)} Section
              </h2>
              <p style={{ color: theme.neutral }}>
                This section is being implemented. The modular structure allows easy addition of new components.
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}