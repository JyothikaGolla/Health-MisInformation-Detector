import React from 'react'

interface RationaleCardProps {
  text: string;
  rationales?: number[];
  tokenizer?: any; // Optional tokenizer for word-level highlighting
}

export default function RationaleCard({ text, rationales }: RationaleCardProps) {
  // If rationales are provided, we can highlight important words
  // For now, we'll just display the text and mention rationale availability
  
  return (
    <div className="border rounded-lg p-3 sm:p-4 shadow-sm bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
      <div className="mb-2">
        <h3 className="text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300">
          Analysis Text
        </h3>
      </div>
      <p className="text-sm sm:text-base text-gray-900 dark:text-gray-100 leading-relaxed break-words">{text}</p>
      
      {rationales && rationales.length > 0 && (
        <div className="mt-2 sm:mt-3 p-2 sm:p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
          <p className="text-xs text-blue-700 dark:text-blue-300">
            ✨ This model identified {rationales.length} rationale scores for different text elements.
            Higher scores indicate more important words for the prediction.
          </p>
        </div>
      )}
    </div>
  )
}
