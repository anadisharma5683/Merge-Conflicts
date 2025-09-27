'use client';

import { Theme } from '@/types/smart-traffic';

interface HeaderProps {
  theme: Theme;
  onLogout: () => void;
}

export default function Header({ theme, onLogout }: HeaderProps) {
  return (
    <header style={{
      background: theme.background,
      padding: '15px 30px',
      borderBottom: `2px solid ${theme.primary}10`,
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      boxShadow: '0 2px 10px rgba(0,0,0,0.05)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
        <div style={{
          width: '40px',
          height: '40px',
          background: `linear-gradient(45deg, ${theme.primary}, ${theme.secondary})`,
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px'
        }}>
          🚦
        </div>
        <h1 style={{ color: theme.primary, margin: 0, fontSize: '20px' }}>
          Smart Traffic Management System
        </h1>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
        <div style={{
          padding: '8px 15px',
          background: `${theme.primary}10`,
          borderRadius: '20px',
          color: theme.primary,
          fontSize: '14px'
        }}>
          Welcome, Admin
        </div>
        <button
          onClick={onLogout}
          style={{
            padding: '8px 15px',
            background: theme.secondary,
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          Logout
        </button>
      </div>
    </header>
  );
}