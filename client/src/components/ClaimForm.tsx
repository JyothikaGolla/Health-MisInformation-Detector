import React, { useState } from 'react'

interface ClaimFormProps {
  onAnalyze: (data: { text: string; model_name: string }) => void;
  loading: boolean;
  models?: string[];
}

export default function ClaimForm({ onAnalyze, loading, models = ["BioBERT", "BioBERT_ARG", "BioBERT_ARG_GNN"] }: ClaimFormProps) {
  const [claim, setClaim] = useState('Turmeric cures cancer')
  const [selectedModel, setSelectedModel] = useState(models[0])

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (claim.trim()) {
      onAnalyze({ text: claim.trim(), model_name: selectedModel })
    }
  }

  return (
    <form onSubmit={submit} className="grid gap-3 p-4 border rounded-lg bg-white dark:bg-gray-800">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Health Claim
        </label>
        <textarea 
          className="w-full border rounded p-3 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 focus:border-blue-500" 
          rows={4} 
          value={claim} 
          onChange={e => setClaim(e.target.value)}
          placeholder="Enter a health claim to analyze..."
          required
        />
      </div>
      
      <div className="flex gap-3 items-center">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Model:
        </label>
        <select 
          className="border rounded px-3 py-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          value={selectedModel}
          onChange={e => setSelectedModel(e.target.value)}
        >
          {models.map(model => (
            <option key={model} value={model}>
              {model.replace(/_/g, '+')}
            </option>
          ))}
        </select>
        
        <button 
          disabled={loading || !claim.trim()} 
          className="rounded px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium transition-colors duration-200 disabled:cursor-not-allowed"
          type="submit"
        >
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
      </div>
    </form>
  )
}
