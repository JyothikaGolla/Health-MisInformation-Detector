import React, { useState } from 'react'

export default function ClaimForm({ onAnalyze, loading }: { onAnalyze: (p: any) => void, loading: boolean }) {
  const [claim, setClaim] = useState('Turmeric cures cancer')
  const [postId, setPostId] = useState('0')

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    onAnalyze({ claim, meta: { postId } })
  }

  return (
    <form onSubmit={submit} className="grid gap-3">
      <textarea className="border rounded p-3" rows={4} value={claim} onChange={e => setClaim(e.target.value)} />
      <div className="flex gap-3 items-center">
        <label className="text-sm opacity-70">Post ID</label>
        <input className="border rounded px-2 py-1 w-24" value={postId} onChange={e => setPostId(e.target.value)} />
        <button disabled={loading} className="rounded px-4 py-2 border">
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
      </div>
    </form>
  )
}
