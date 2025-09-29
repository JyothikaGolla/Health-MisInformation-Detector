// App.tsx
import React, { useState } from "react";
import { SunIcon, MoonIcon, InformationCircleIcon } from '@heroicons/react/24/outline';
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";

interface PredictionResult {
  probs: number[][];
  pred: number;
  rationales?: number[][];
}

interface ModelResult {
  model: string;
  prediction: "Real" | "Fake";
  confidence: number;
  rationales?: number[];
}

interface CompareResult {
  input: string;
  results: ModelResult[];
}

const App: React.FC = () => {
  const [claim, setClaim] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const [analyzeModel, setAnalyzeModel] = useState("BioBERT");
  const [singleResult, setSingleResult] = useState<ModelResult | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const models = ["BioBERT", "BioBERT_ARG", "BioBERT_ARG_GNN"];
  
  // Sample health claims from the dataset
  const sampleClaims = [
    "BCG vaccine leads to long-term blood sugar improvement in type 1 diabetes patients",
    "Drinking lemon water in the morning boosts metabolism and helps with weight loss",
    "5G cell towers cause COVID-19 symptoms and weaken the immune system",
    "Taking vitamin C supplements prevents the common cold completely",
    "Magic mushrooms can cure depression and anxiety disorders permanently"
  ];

  const handleAnalyze = async () => {
    if (!claim.trim()) return;
    setLoading(true);
    setError("");
    setSingleResult(null);
    setCompareResult(null);
    
    try {
      const response = await axios.post<PredictionResult>("http://127.0.0.1:8000/predict", {
        text: claim,
        model_type: analyzeModel,
      });
      
      const result: ModelResult = {
        model: analyzeModel,
        prediction: response.data.pred === 0 ? "Fake" : "Real",
        confidence: Math.max(...response.data.probs[0]) * 100,
        rationales: response.data.rationales?.[0]
      };
      
      setSingleResult(result);
    } catch (err: any) {
      setError(err.response?.data?.error || "Failed to analyze claim");
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (!claim.trim()) return;
    setLoading(true);
    setError("");
    setSingleResult(null);
    setCompareResult(null);
    
    try {
      const promises = models.map(model => 
        axios.post<PredictionResult>("http://127.0.0.1:8000/predict", {
          text: claim,
          model_type: model,
        })
      );
      
      const responses = await Promise.all(promises);
      
      const results: ModelResult[] = responses.map((response, index) => ({
        model: models[index],
        prediction: response.data.pred === 0 ? "Fake" : "Real",
        confidence: Math.max(...response.data.probs[0]) * 100,
        rationales: response.data.rationales?.[0]
      }));
      
      setCompareResult({
        input: claim,
        results
      });
    } catch (err: any) {
      setError(err.response?.data?.error || "Failed to compare models");
    } finally {
      setLoading(false);
    }
  };

  const handleSampleClaim = (sampleClaim: string) => {
    setClaim(sampleClaim);
  };

  const getChartData = () => {
    if (compareResult) {
      return compareResult.results.map(result => ({
        model: result.model,
        confidence: result.confidence.toFixed(1),
        prediction: result.prediction
      }));
    }
    return [];
  };

  return (
    <div className={
      `${darkMode ? "dark bg-gradient-to-br from-gray-900 to-gray-800" : "bg-gradient-to-br from-blue-50 to-indigo-100"} min-h-screen transition-colors duration-300`
    }>
      {/* Sticky Header */}
      <header className="sticky top-0 z-20 bg-white/80 dark:bg-gray-900/80 backdrop-blur border-b border-gray-200 dark:border-gray-800 shadow-sm flex items-center justify-between px-6 py-3">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-bold text-indigo-700 dark:text-indigo-300">🔍 Health Misinfo Detector</span>
        </div>
        <div className="flex items-center gap-4">
          <button
            aria-label={darkMode ? "Switch to light mode" : "Switch to dark mode"}
            className="rounded-full p-2 hover:bg-indigo-100 dark:hover:bg-gray-700 transition"
            onClick={() => setDarkMode((d) => !d)}
          >
            {darkMode ? <SunIcon className="h-6 w-6 text-yellow-400" /> : <MoonIcon className="h-6 w-6 text-gray-700" />}
          </button>
        </div>
      </header>
      <div className="max-w-6xl mx-auto px-2 sm:px-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 dark:text-indigo-100 mb-4">
            Analyze health claims using advanced AI models including BioBERT with Argument Mining and Graph Neural Networks
          </h1>
          <div className="flex flex-col md:flex-row items-center justify-center gap-6 mt-4">
            <div className="flex-shrink-0 flex items-center justify-center">
              <InformationCircleIcon className="h-16 w-16 md:h-24 md:w-24 text-blue-400 dark:text-blue-300" />
            </div>
            <div className="max-w-md text-lg text-gray-600 dark:text-gray-300 text-left">
              <span>Choose a claim and model to get started.<br />
              <span className="text-sm text-gray-500 dark:text-gray-400">Results are AI predictions, not medical advice.</span></span>
            </div>
          </div>
        </div>

        {/* Sample Claims */}
  <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Try Sample Health Claims:</h2>
          <div className="flex flex-wrap gap-2">
            {sampleClaims.map((sampleClaim, index) => (
              <button
                key={index}
                className="text-left px-3 py-2 bg-gray-50 dark:bg-gray-800 hover:bg-blue-100 dark:hover:bg-indigo-900 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-300 transition-all duration-200 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
                onClick={() => handleSampleClaim(sampleClaim)}
                tabIndex={0}
                aria-label={`Try sample claim: ${sampleClaim}`}
              >
                {sampleClaim}
              </button>
            ))}
          </div>
        </div>

        {/* Input Section */}
  <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Enter Health Claim:</h2>
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center mt-2">
            <textarea
              className="w-full p-4 border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100 rounded-lg mb-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none shadow"
              rows={4}
              placeholder="Enter a health claim to analyze (e.g., 'Vitamin C prevents COVID-19', 'Coffee causes cancer', etc.)..."
              value={claim}
              onChange={(e) => setClaim(e.target.value)}
              aria-label="Health claim input"
            />
            <button
              className="bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 px-4 py-2 rounded-lg font-medium transition-colors duration-200 hover:bg-gray-300 dark:hover:bg-gray-600 mb-2 shadow"
              onClick={() => setClaim("")}
              disabled={!claim.trim()}
              aria-label="Clear claim input"
            >Clear</button>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
            <div className="flex items-center gap-2 mt-2">
              <label htmlFor="model-select" className="text-sm font-medium text-gray-700 dark:text-gray-200">
                Model:
              </label>
              <select
                id="model-select"
                className="border border-gray-300 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow"
                value={analyzeModel}
                onChange={(e) => setAnalyzeModel(e.target.value)}
                aria-label="Model selection"
              >
                {models.map((model) => (
                  <option key={model} value={model}>
                    {model.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
              <span className="ml-1" title="Select which AI model to use for analysis."><InformationCircleIcon className="h-4 w-4 text-blue-400 dark:text-blue-300" /></span>
            </div>

            <div className="flex gap-3">
              <button
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow focus:outline-none focus:ring-2 focus:ring-blue-400"
                onClick={handleAnalyze}
                disabled={loading || !claim.trim()}
                aria-label="Analyze claim"
              >
                {loading ? (
                  <span className="flex items-center gap-2"><span className="animate-spin h-5 w-5 border-b-2 border-white rounded-full"></span>Analyzing...</span>
                ) : "🔍 Analyze"}
              </button>

              <button
                className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg font-medium transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow focus:outline-none focus:ring-2 focus:ring-green-400"
                onClick={handleCompare}
                disabled={loading || !claim.trim()}
                aria-label="Compare models"
              >
                {loading ? (
                  <span className="flex items-center gap-2"><span className="animate-spin h-5 w-5 border-b-2 border-white rounded-full"></span>Comparing...</span>
                ) : "⚖️ Compare Models"}
              </button>
            </div>
          </div>
        </div>

        {/* Loading and Error States */}
        {loading && (
          <div className="bg-blue-50 dark:bg-gray-800 border-l-4 border-blue-400 dark:border-blue-600 p-4 mb-6 rounded animate-pulse">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 dark:border-blue-300 mr-3"></div>
              <p className="text-blue-700 dark:text-blue-200 font-medium">Processing your health claim...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 dark:bg-red-900 border-l-4 border-red-400 dark:border-red-700 p-4 mb-6 rounded animate-shake">
            <p className="text-red-700 dark:text-red-200 font-medium">❌ Error: {error}</p>
          </div>
        )}

        {/* Single Model Result */}
        {singleResult && (
          <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6 mb-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              📊 Analysis Result - {singleResult.model.replace(/_/g, " ")}
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Prediction:</p>
                  <p className={`text-2xl font-bold ${
                    singleResult.prediction === "Real" ? "text-green-600" : "text-red-600"
                  }`}>
                    {singleResult.prediction === "Real" ? "✅ Real" : "❌ Fake"}
                  </p>
                </div>
                
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Confidence:</p>
                  <div className="flex items-center">
                    <div className="flex-1 bg-gray-200 rounded-full h-3 mr-3">
                      <div 
                        className={`h-3 rounded-full ${
                          singleResult.confidence > 70 ? "bg-green-500" : 
                          singleResult.confidence > 50 ? "bg-yellow-500" : "bg-red-500"
                        }`}
                        style={{ width: `${singleResult.confidence}%` }}
                      ></div>
                    </div>
                    <span className="text-lg font-bold">
                      {singleResult.confidence.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-2">Input Claim:</p>
                <p className="text-gray-800 dark:text-gray-100 italic">"{claim}"</p>
              </div>
            </div>
            
            {singleResult.rationales && singleResult.rationales.length > 0 && (
              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900 rounded-lg">
                <p className="text-sm font-medium text-blue-700 dark:text-blue-200 mb-2">
                  🎯 Argument Rationales <span title="Words most important for the model's decision are highlighted."><InformationCircleIcon className="h-4 w-4 inline text-blue-400 dark:text-blue-300" /></span>
                </p>
                <p className="text-sm text-blue-600 dark:text-blue-200">
                  This model identified key argumentative elements in the text to make its prediction.
                  Higher rationale scores indicate more important words for the decision.
                </p>
              </div>
            )}
          </div>
        )}

        {/* Compare Results */}
        {compareResult && (
          <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6 mb-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              ⚖️ Model Comparison Results
            </h2>
            
            <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm font-medium text-gray-600 mb-1">Analyzed Claim:</p>
              <p className="text-gray-800 italic">"{compareResult.input}"</p>
            </div>

            {/* Chart Visualization */}
            <div className="mb-8">
              <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-4">Confidence Comparison</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={getChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="model" 
                      tick={{ fontSize: 12 }}
                      angle={-45}
                      textAnchor="end"
                      height={80}
                    />
                    <YAxis 
                      domain={[0, 100]}
                      tick={{ fontSize: 12 }}
                      label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      formatter={(value) => [`${value}%`, 'Confidence']}
                      labelFormatter={(label) => `Model: ${label.replace(/_/g, ' ')}`}
                    />
                    <Legend />
                    <Bar dataKey="confidence" name="Confidence">
                      {getChartData().map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.prediction === "Real" ? "#10B981" : "#EF4444"} 
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Detailed Results Table */}
            <div className="overflow-x-auto">
              <table className="w-full border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                <thead className="bg-gray-100 dark:bg-gray-800">
                  <tr>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-200">Model</th>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-200">Prediction</th>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-200">Confidence</th>
                    <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-200">Features</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {compareResult.results.map((result, index) => (
                    <tr key={result.model} className={index % 2 === 0 ? "bg-white dark:bg-gray-900" : "bg-gray-50 dark:bg-gray-800"}>
                      <td className="px-6 py-4 text-sm font-medium text-gray-900 dark:text-gray-100">
                        {result.model.replace(/_/g, " ")}
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                          result.prediction === "Real" 
                            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" 
                            : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                        }`}>
                          {result.prediction === "Real" ? "✅ Real" : "❌ Fake"}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-700 dark:text-gray-200">
                        <div className="flex items-center">
                          <div className="flex-1 bg-gray-200 rounded-full h-2 mr-3 w-20">
                            <div 
                              className={`h-2 rounded-full ${
                                result.confidence > 70 ? "bg-green-500" : 
                                result.confidence > 50 ? "bg-yellow-500" : "bg-red-500"
                              }`}
                              style={{ width: `${result.confidence}%` }}
                            ></div>
                          </div>
                          <span className="font-medium">
                            {result.confidence.toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-700 dark:text-gray-200">
                        {result.model === "BioBERT" && "Base BioBERT model"}
                        {result.model === "BioBERT_ARG" && "BioBERT + Argument Mining"}
                        {result.model === "BioBERT_ARG_GNN" && "BioBERT + Arguments + Graph Neural Network"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Model Information */}
  <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-semibold text-gray-800 dark:text-indigo-100 mb-4">🧠 About the Models</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <h3 className="font-semibold text-blue-600 dark:text-blue-300 mb-2">BioBERT</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Base biomedical language model trained on PubMed abstracts and full-text papers.
                Specializes in understanding medical and health-related terminology.
              </p>
            </div>
            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <h3 className="font-semibold text-green-600 dark:text-green-300 mb-2">BioBERT + ARG</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Enhanced with argument mining capabilities to identify and weigh important 
                argumentative elements in health claims for better reasoning.
              </p>
            </div>
            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <h3 className="font-semibold text-purple-600 dark:text-purple-300 mb-2">BioBERT + ARG + GNN</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Most advanced model combining BioBERT, argument mining, and graph neural networks
                to understand relationships between concepts in medical claims.
              </p>
            </div>
          </div>
        </div>
      </div>
      {/* Footer */}
      <footer className="mt-12 py-6 text-center text-gray-500 dark:text-gray-400 text-sm border-t border-gray-200 dark:border-gray-800">
        <span>© {new Date().getFullYear()} Health Misinfo Detector. Built with React, Tailwind CSS, and FastAPI. | <a href="https://github.com/JyothikaGolla/Health-MisInformation-Detector" className="underline hover:text-indigo-600 dark:hover:text-indigo-300">GitHub</a></span>
      </footer>
    </div>
  );
};

export default App;
