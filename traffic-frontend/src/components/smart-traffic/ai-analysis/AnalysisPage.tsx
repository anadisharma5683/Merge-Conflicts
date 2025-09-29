'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { initialAreas } from '@/lib/data';
import { BrainCircuit } from 'lucide-react';

// 🎲 Function to generate random pie chart data that sums to 100
const generateCongestionData = () => {
  const values = [0, 0, 0, 0];
  let remaining = 100;

  for (let i = 0; i < 3; i++) {
    const val = Math.floor(Math.random() * (remaining / 2)) + 10; // keep values spread
    values[i] = val;
    remaining -= val;
  }
  values[3] = remaining; // last one gets the rest

  return [
    { name: 'Critical', value: values[0] },
    { name: 'High', value: values[1] },
    { name: 'Medium', value: values[2] },
    { name: 'Low', value: values[3] },
  ];
};

const congestionStatusData = generateCongestionData();

const COLORS = {
  Critical: '#ef4444',
  High: '#f97316',
  Medium: '#f59e0b',
  Low: '#10b981',
};

const vehicleTrendData = [
  { name: '06:00', 'rajmahal-square': 150, 'kalpana-square': 100, 'shastri-nagar': 50, 'acharya-vihar': 30 },
  { name: '09:00', 'rajmahal-square': 350, 'kalpana-square': 280, 'shastri-nagar': 150, 'acharya-vihar': 120 },
  { name: '12:00', 'rajmahal-square': 400, 'kalpana-square': 320, 'shastri-nagar': 180, 'acharya-vihar': 140 },
  { name: '15:00', 'rajmahal-square': 380, 'kalpana-square': 300, 'shastri-nagar': 160, 'acharya-vihar': 130 },
  { name: '18:00', 'rajmahal-square': 450, 'kalpana-square': 350, 'shastri-nagar': 200, 'acharya-vihar': 160 },
  { name: '21:00', 'rajmahal-square': 300, 'kalpana-square': 250, 'shastri-nagar': 120, 'acharya-vihar': 80 },
];

export default function AnalysisPage() {
  return (
    <div className="p-4 md:p-6 lg:p-8">
      <h1 className="text-3xl font-bold tracking-tight mb-6 flex items-center gap-2">
        <BrainCircuit className="size-8" /> AI Analysis
      </h1>
      <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
        {/* Congestion Level by Area */}
        <Card>
          <CardHeader>
            <CardTitle>Congestion Level by Area</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={initialAreas}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="congestionLevel" fill="#2962FF" name="Congestion Level (%)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Randomized Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Congestion Status Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={congestionStatusData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={120}
                  innerRadius={60} // donut style
                  label={({ name, value }) => `${name} - ${value}%`}
                >
                  {congestionStatusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[entry.name as keyof typeof COLORS]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Vehicle Trends */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Vehicle Count Trends (Today)</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={vehicleTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="rajmahal-square" stroke="#ef4444" />
                <Line type="monotone" dataKey="kalpana-square" stroke="#f97316" />
                <Line type="monotone" dataKey="shastri-nagar" stroke="#f59e0b" />
                <Line type="monotone" dataKey="acharya-vihar" stroke="#10b981" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Average Speed */}
        <Card>
          <CardHeader>
            <CardTitle>Average Speed by Area</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={initialAreas} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={100} />
                <Tooltip />
                <Legend />
                <Bar dataKey="averageSpeed" fill="#A044FF" name="Average Speed (km/h)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Average Wait Time */}
        <Card>
          <CardHeader>
            <CardTitle>Average Wait Time by Area</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={initialAreas}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="waitTime" fill="#2962FF" name="Wait Time (min)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
