import SmartTrafficSystem from '@/components/smart-traffic';
import ErrorBoundary from '@/components/ErrorBoundary';

export default function HomePage() {
  return (
    <ErrorBoundary>
      <div suppressHydrationWarning={true}>
        <SmartTrafficSystem />
      </div>
    </ErrorBoundary>
  );
}