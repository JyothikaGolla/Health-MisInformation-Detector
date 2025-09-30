import React from 'react'

interface ScoreBadgeProps {
  verdict: string;
  confidence: number;
}

export default function ScoreBadge({ verdict, confidence }: ScoreBadgeProps) {
  // Map backend labels to colors - backend returns "reliable" or "misinformation"
  const getColor = (label: string) => {
    switch (label.toLowerCase()) {
      case 'reliable':
        return '#22C55E' // Bright green
      case 'misinformation':
        return '#EF4444' // Bright red
      case 'real':
        return '#22C55E' // Bright green
      case 'fake':
        return '#EF4444' // Bright red
      default:
        return '#6B7280' // Gray
    }
  }

  const color = getColor(verdict)
  
  return (
    <div className={`inline-flex items-center gap-1 sm:gap-2 border rounded px-2 sm:px-3 py-1 bg-gray-50 dark:bg-gray-800`}>
      <span className={`w-2 h-2 rounded-full`} style={{ background: color }}></span>
      <span className="text-sm sm:text-base font-semibold capitalize">{verdict}</span>
      <span className="opacity-70 text-xs sm:text-sm">({Math.round(confidence * 100)}%)</span>
    </div>
  )
}
