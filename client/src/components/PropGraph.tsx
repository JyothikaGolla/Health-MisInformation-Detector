import React from 'react'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts'

export default function PropGraph({ cues }: { cues?: any }) {
  const data = [
    { name: 'Shares', value: cues?.shares ?? 0 },
    { name: 'Influencers', value: cues?.influencers ?? 0 }
  ]
  return (
    <div className="border rounded p-3">
      <div className="h-60">
        <ResponsiveContainer>
          <BarChart data={data}>
            <XAxis dataKey="name" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="value" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
