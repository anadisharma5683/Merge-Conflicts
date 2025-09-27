'use client';

import { 
  MapPin, 
  Play, 
  Settings, 
  AlertTriangle, 
  Activity,
  BarChart3
} from 'lucide-react';
import { Theme, NavItem } from '@/types/smart-traffic';

interface SidebarProps {
  activeSection: string;
  theme: Theme;
  onSectionChange: (section: string) => void;
}

export default function Sidebar({ activeSection, theme, onSectionChange }: SidebarProps) {
  const navItems: NavItem[] = [
    { id: 'map', icon: MapPin, label: 'Interactive Map' },
    { id: 'video', icon: Play, label: 'Live Video Feed' },
    { id: 'signals', icon: Settings, label: 'Signal Status' },
    { id: 'congestion', icon: Activity, label: 'Congestion Monitor' },
    { id: 'analytics', icon: BarChart3, label: 'Traffic Analytics' },
    { id: 'accidents', icon: AlertTriangle, label: 'Accident Reports' }
  ];

  return (
    <nav style={{
      width: '250px',
      background: theme.background,
      borderRight: `2px solid ${theme.primary}10`,
      padding: '20px'
    }}>
      {navItems.map(item => (
        <button
          key={item.id}
          onClick={() => onSectionChange(item.id)}
          style={{
            width: '100%',
            padding: '15px',
            background: activeSection === item.id ? `${theme.primary}15` : 'transparent',
            border: activeSection === item.id ? `2px solid ${theme.primary}` : 'none',
            borderRadius: '8px',
            color: activeSection === item.id ? theme.primary : theme.darkText,
            cursor: 'pointer',
            marginBottom: '10px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            fontSize: '14px',
            textAlign: 'left'
          }}
        >
          <item.icon size={18} />
          {item.label}
        </button>
      ))}
    </nav>
  );
}