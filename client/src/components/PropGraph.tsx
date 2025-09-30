import React from 'react'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell } from 'recharts'

interface PropGraphProps {
  data?: {
    probabilities?: {
      misinformation: number;
      reliable: number;
    };
    confidence?: number;
  };
  type?: 'bar' | 'pie';
}

export default function PropGraph({ data, type = 'pie' }: PropGraphProps) {
  if (!data?.probabilities) {
    return (
      <div className="border rounded p-3 bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700">
        <p className="text-gray-500 dark:text-gray-400 text-center">No probability data available</p>
      </div>
    )
  }

  const chartData = [
    { 
      name: 'Misinformation', 
      value: Math.round(data.probabilities.misinformation * 100),
      fullValue: data.probabilities.misinformation
    },
    { 
      name: 'Reliable', 
      value: Math.round(data.probabilities.reliable * 100),
      fullValue: data.probabilities.reliable
    }
  ]

  const colors = ['#EF4444', '#22C55E'] // Brighter red for misinformation, Brighter green for reliable

  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, name, value }: any) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 1.4;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    // Adjust positioning for better visibility on mobile
    let adjustedX = x;
    if (x < cx && name === 'Misinformation') {
      // For misinformation label on the left, move it further right
      adjustedX = Math.max(x, 15);
    }

    return (
      <text 
        x={adjustedX} 
        y={y} 
        fill="#E5E7EB" 
        textAnchor={adjustedX > cx ? 'start' : 'end'} 
        dominantBaseline="central"
        fontSize={9}
        fontWeight="500"
        className="text-xs sm:text-sm"
      >
        <tspan className="hidden sm:inline">{`${name}: ${value}%`}</tspan>
        <tspan className="sm:hidden">{`${value}%`}</tspan>
      </text>
    );
  };

  if (type === 'pie') {
    return (
      <div className="border rounded-lg p-3 sm:p-4 md:p-6 lg:p-3 bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
        <h3 className="text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 sm:mb-3 text-center">
          Probability Distribution
        </h3>
        <div className="px-2 sm:px-3 md:px-3 lg:px-7 h-40 sm:h-44 md:h-48" style={{ width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={15}
                outerRadius={30}
                dataKey="value"
                label={renderCustomLabel}
                labelLine={false}
              >
                {chartData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={colors[index]}
                    stroke="#374151"
                    strokeWidth={1}
                  />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value) => `${value}%`}
                contentStyle={{ 
                  fontSize: '12px',
                  backgroundColor: '#374151',
                  border: '1px solid #6B7280',
                  borderRadius: '6px',
                  color: '#F3F4F6'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        
        {/* Mobile-friendly legend */}
        <div className="flex justify-center gap-3 sm:gap-4 mt-2 sm:mt-3 text-xs sm:text-sm">
          <div className="flex items-center gap-1 sm:gap-2">
            <div className="w-2 h-2 sm:w-3 sm:h-3 rounded-full" style={{ backgroundColor: colors[0] }}></div>
            <span className="text-gray-700 dark:text-gray-300">
              <span className="hidden sm:inline">Misinformation: </span>
              <span className="sm:hidden">Misinfo: </span>
              {chartData[0].value}%
            </span>
          </div>
          <div className="flex items-center gap-1 sm:gap-2">
            <div className="w-2 h-2 sm:w-3 sm:h-3 rounded-full" style={{ backgroundColor: colors[1] }}></div>
            <span className="text-gray-700 dark:text-gray-300">
              <span className="hidden sm:inline">Reliable: </span>
              <span className="sm:hidden">Reliable: </span>
              {chartData[1].value}%
            </span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="border rounded-lg p-3 sm:p-4 bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
      <h3 className="text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 sm:mb-3">
        Model Confidence
      </h3>
      <div className="h-48 sm:h-60">
        <ResponsiveContainer>
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
            <XAxis 
              dataKey="name" 
              tick={{ fontSize: 10 }}
              className="text-xs sm:text-sm"
            />
            <YAxis 
              domain={[0, 100]} 
              tick={{ fontSize: 10 }}
              className="text-xs sm:text-sm"
            />
            <Tooltip 
              formatter={(value) => `${value}%`}
              contentStyle={{ 
                fontSize: '12px',
                backgroundColor: '#374151',
                border: '1px solid #6B7280',
                borderRadius: '6px',
                color: '#F3F4F6'
              }}
            />
            <Bar dataKey="value">
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
