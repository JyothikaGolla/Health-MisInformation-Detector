import React from 'react'

export default function RationaleCard({ text }: { text: string }) {
  return (
    <div className="border rounded p-3 shadow-sm">
      <p>{text}</p>
    </div>
  )
}
