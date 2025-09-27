'use client';

import { Theme } from '@/types/smart-traffic';

interface LoginScreenProps {
  username: string;
  password: string;
  loginError: string;
  theme: Theme;
  onUsernameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onLogin: (e: { preventDefault: () => void; }) => void;
}

export default function LoginScreen({
  username,
  password,
  loginError,
  theme,
  onUsernameChange,
  onPasswordChange,
  onLogin
}: LoginScreenProps) {
  return (
    <div style={{
      minHeight: '100vh',
      background: `linear-gradient(135deg, ${theme.primary} 0%, ${theme.secondary} 100%)`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      <div style={{
        background: theme.background,
        padding: '40px',
        borderRadius: '15px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.1)',
        width: '100%',
        maxWidth: '400px'
      }}>
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          <div style={{
            width: '80px',
            height: '80px',
            background: `linear-gradient(45deg, ${theme.primary}, ${theme.secondary})`,
            borderRadius: '50%',
            margin: '0 auto 20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '32px'
          }}>
            🚦
          </div>
          <h1 style={{ color: theme.primary, margin: '0', fontSize: '24px' }}>
            Smart Traffic Management
          </h1>
          <p style={{ color: theme.neutral, margin: '10px 0 0 0' }}>
            Login to access the system
          </p>
        </div>

        <form onSubmit={onLogin}>
          <div style={{ marginBottom: '20px' }}>
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => onUsernameChange(e.target.value)}
              style={{
                width: '100%',
                padding: '15px',
                border: `2px solid ${theme.primary}20`,
                borderRadius: '8px',
                fontSize: '16px',
                boxSizing: 'border-box'
              }}
              required
            />
          </div>

          <div style={{ marginBottom: '20px' }}>
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => onPasswordChange(e.target.value)}
              style={{
                width: '100%',
                padding: '15px',
                border: `2px solid ${theme.primary}20`,
                borderRadius: '8px',
                fontSize: '16px',
                boxSizing: 'border-box'
              }}
              required
            />
          </div>

          {loginError && (
            <div style={{
              color: theme.secondary,
              background: `${theme.secondary}10`,
              padding: '10px',
              borderRadius: '5px',
              marginBottom: '20px',
              fontSize: '14px'
            }}>
              {loginError}
            </div>
          )}

          <button
            type="submit"
            style={{
              width: '100%',
              padding: '15px',
              background: `linear-gradient(45deg, ${theme.primary}, ${theme.secondary})`,
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '16px',
              fontWeight: 'bold',
              cursor: 'pointer'
            }}
          >
            Login to Dashboard
          </button>
        </form>

        <div style={{
          marginTop: '30px',
          padding: '15px',
          background: theme.accent,
          borderRadius: '8px',
          fontSize: '14px',
          color: theme.neutral
        }}>
          <strong>Demo Credentials:</strong><br />
          Username: admin<br />
          Password: admin<br/>
          <strong>For :- ODISHA GOVERNMENT</strong>
        </div>
      </div>
    </div>
  );
}