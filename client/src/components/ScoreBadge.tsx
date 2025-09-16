import React from 'react'

export default function ScoreBadge({ verdict, confidence }: { verdict: string, confidence: number }) {
  const color = verdict === 'true' ? 'green' : verdict === 'fake' ? 'red' : 'gray'
  return (
    <div className={`inline-flex items-center gap-2 border rounded px-3 py-1`}>
      <span className={`w-2 h-2 rounded-full`} style={{ background: color }}></span>
      <span className="font-semibold capitalize">{verdict}</span>
      <span className="opacity-70 text-sm">({(confidence*100).toFixed(0)}%)</span>
    </div>
  )
}
