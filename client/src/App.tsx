import React, { useState } from "react";
import { InformationCircleIcon } from '@heroicons/react/24/outline';
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import PropGraph from "./components/PropGraph";
import RationaleCard from "./components/RationaleCard";
import ScoreBadge from "./components/ScoreBadge";
import { PredictionResult, ModelResult, CompareResult, PredictionRequest } from "./types/api";
import { API_BASE_URL } from "./config/api";

const App: React.FC = () => {
  const [claim, setClaim] = useState("");
  const [analyzeModel, setAnalyzeModel] = useState("BioBERT");
  const [singleResult, setSingleResult] = useState<ModelResult | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const models = ["BioBERT", "BioBERT_ARG", "BioBERT_ARG_GNN"];
  
  const sampleClaims = [
    "BCG vaccine leads to long-term blood sugar improvement in type 1 diabetes patients",
    "Drinking lemon water in the morning boosts metabolism and helps with weight loss",
    "5G cell towers cause COVID-19 symptoms and weaken the immune system",
    "Taking vitamin C supplements prevents the common cold completely",
    "Magic mushrooms can cure depression and anxiety disorders permanently"
  ];

  const formatModelName = (modelName: string) => {
    return modelName.replace(/_/g, "+");
  };

  const handleAnalyze = async () => {
    if (!claim.trim()) return;
    console.log("Starting analysis with:", { claim, analyzeModel }); // Debug log
    setLoading(true);
    setError("");
    setSingleResult(null);
    setCompareResult(null);
    
    try {
      console.log("Making request to backend..."); // Debug log
      console.log("API URL:", `${API_BASE_URL}/predict`); // Debug log
      console.log("Request payload:", { text: claim, model_name: analyzeModel }); // Debug log
      
      const response = await axios.post<PredictionResult>(`${API_BASE_URL}/predict`, {
        text: claim,
        model_name: analyzeModel,
      }, {
        timeout: 30000, // 30 second timeout
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      console.log("API Response received:", response.data); // Debug log
      console.log("Response status:", response.status); // Debug log
      
      // Validate response structure
      if (!response.data || typeof response.data.confidence !== 'number' || !response.data.label) {
        throw new Error(`Invalid response format: ${JSON.stringify(response.data)}`);
      }
      
      const result: ModelResult = {
        model: analyzeModel,
        prediction: response.data.label === "reliable" ? "Real" : "Fake",
        confidence: response.data.confidence * 100,
        label: response.data.label,
        rationales: response.data.rationales && response.data.rationales.length > 0 ? response.data.rationales.flat() : undefined
      };
      
      console.log("Processed result:", result); // Debug log
      setSingleResult(result);
    } catch (err: any) {
      console.error("Analysis error occurred:", err);
      console.error("Error response:", err.response);
      console.error("Error status:", err.response?.status);
      console.error("Error data:", err.response?.data);
      setError(err.response?.data?.error || err.message || "Failed to analyze claim. Please check if the backend is running.");
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
      const promises = models.map(async (model) => {
        const response = await axios.post<PredictionResult>(`${API_BASE_URL}/predict`, {
          text: claim,
          model_name: model,
        }, {
          timeout: 30000, // 30 second timeout
          headers: {
            'Content-Type': 'application/json',
          }
        });
        
        return {
          model,
          prediction: response.data.label === "reliable" ? "Real" : "Fake" as "Real" | "Fake",
          confidence: response.data.confidence * 100,
          label: response.data.label,
          rationales: response.data.rationales && response.data.rationales.length > 0 ? response.data.rationales.flat() : undefined
        };
      });
      
      const results = await Promise.all(promises);
      
      setCompareResult({
        input: claim,
        results
      });
    } catch (err: any) {
      console.error("Comparison error:", err);
      console.error("Error details:", err.response); // More detailed error logging
      setError(err.response?.data?.error || err.message || "Failed to compare models. Please check if the backend is running.");
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
        model: formatModelName(result.model),
        confidence: result.confidence.toFixed(1),
        prediction: result.prediction
      }));
    }
    return [];
  };

  return (
    <div className="dark bg-gradient-to-br from-gray-900 to-gray-800 min-h-screen">
      <header className="sticky top-0 z-20 bg-gray-900/80 backdrop-blur border-b border-gray-800 shadow-sm flex items-center justify-between px-3 sm:px-6 py-3">
        <div className="flex items-center gap-2 sm:gap-3">
          <img 
            src="./logo.PNG" 
            alt="Health Misinformation Detector Logo" 
            className="h-6 w-6 sm:h-8 sm:w-8"
          />
          <span className="text-lg sm:text-xl md:text-2xl font-bold text-indigo-300">Health Misinformation Detector</span>
        </div>
      </header>
      
      <div className="max-w-6xl mx-auto px-3 sm:px-6">
        <div className="text-center mb-6 sm:mb-8 py-4 sm:py-8">
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-indigo-100 mb-4 sm:mb-6 px-2">
            Analyze health claims using advanced AI models including BioBERT with Adaptive Rationale Guidance and Graph Neural Networks
          </h1>
          <div className="flex flex-col items-center justify-center gap-3 sm:gap-4">
            <div className="flex-shrink-0 flex items-center justify-center">
              <InformationCircleIcon className="h-12 w-12 sm:h-16 sm:w-16 md:h-24 md:w-24 text-blue-300" />
            </div>
            <div className="max-w-md text-base sm:text-lg text-gray-300 text-center px-4">
              <span>Choose a claim and model to get started.<br />
              <span className="text-sm text-gray-400">Results are AI predictions, not medical advice.</span></span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 mb-4 sm:mb-6">
          <h2 className="text-lg sm:text-xl font-semibold text-white mb-3 sm:mb-4">Try Sample Health Claims:</h2>
          <div className="flex flex-wrap gap-2">
            {sampleClaims.map((sampleClaim, index) => (
              <button
                key={index}
                className="text-left px-2 sm:px-3 py-2 bg-gray-700 hover:bg-blue-800 rounded-lg border border-gray-600 hover:border-blue-500 transition-all duration-200 text-xs sm:text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 text-gray-200 break-words"
                onClick={() => handleSampleClaim(sampleClaim)}
                tabIndex={0}
                aria-label={`Try sample claim: ${sampleClaim}`}
              >
                {sampleClaim}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 mb-4 sm:mb-6">
          <h2 className="text-lg sm:text-xl font-semibold text-white mb-3 sm:mb-4">Enter Health Claim:</h2>
          
          <div className="flex flex-col gap-3 sm:gap-4 items-start mt-2">
            <div className="w-full flex flex-col sm:flex-row gap-2 sm:gap-4 items-start">
              <textarea
                className="w-full p-3 sm:p-4 border border-gray-600 bg-gray-700 text-white rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none shadow placeholder-gray-300 text-sm sm:text-base"
                rows={3}
                placeholder="Enter a health claim to analyze (e.g., 'Vitamin C prevents COVID-19', 'Coffee causes cancer', etc.)..."
                value={claim}
                onChange={(e) => setClaim(e.target.value)}
                aria-label="Health claim input"
              />
              <button
                className="w-full sm:w-auto bg-gray-600 text-gray-200 px-3 sm:px-4 py-2 rounded-lg font-medium transition-colors duration-200 hover:bg-gray-500 shadow text-sm sm:text-base whitespace-nowrap"
                onClick={() => setClaim("")}
                disabled={!claim.trim()}
                aria-label="Clear claim input"
              >Clear</button>
            </div>

            <div className="w-full flex flex-col sm:flex-row gap-3 sm:gap-4 items-start">
              <div className="flex items-center gap-2 w-full sm:w-auto">
                <label htmlFor="model-select" className="text-sm font-medium text-gray-200 whitespace-nowrap">
                  Model:
                </label>
                <select
                  id="model-select"
                  className="flex-1 sm:flex-none border border-gray-600 bg-gray-700 text-white rounded-lg px-2 sm:px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow text-sm sm:text-base"
                  value={analyzeModel}
                  onChange={(e) => setAnalyzeModel(e.target.value)}
                  aria-label="Model selection"
                >
                  {models.map((model) => (
                    <option key={model} value={model}>
                      {formatModelName(model)}
                    </option>
                  ))}
                </select>
                <span className="ml-1" title="Select which AI model to use for analysis.">
                  <InformationCircleIcon className="h-4 w-4 text-blue-300" />
                </span>
              </div>

              <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 w-full sm:w-auto">
                <button
                  className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700 text-white px-4 sm:px-6 py-2 rounded-lg font-medium transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow focus:outline-none focus:ring-2 focus:ring-blue-400 text-sm sm:text-base"
                  onClick={handleAnalyze}
                  disabled={loading || !claim.trim()}
                  aria-label="Analyze claim"
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <span className="animate-spin h-4 w-4 sm:h-5 sm:w-5 border-b-2 border-white rounded-full"></span>
                      <span className="hidden sm:inline">Analyzing...</span>
                      <span className="sm:hidden">...</span>
                    </span>
                  ) : (
                    <span className="flex items-center justify-center gap-1 sm:gap-2">
                      <span>🔍</span>
                      <span className="hidden sm:inline">Analyze</span>
                      <span className="sm:hidden">Analyze</span>
                    </span>
                  )}
                </button>

                <button
                  className="w-full sm:w-auto bg-green-600 hover:bg-green-700 text-white px-4 sm:px-6 py-2 rounded-lg font-medium transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow focus:outline-none focus:ring-2 focus:ring-green-400 text-sm sm:text-base"
                  onClick={handleCompare}
                  disabled={loading || !claim.trim()}
                  aria-label="Compare models"
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <span className="animate-spin h-4 w-4 sm:h-5 sm:w-5 border-b-2 border-white rounded-full"></span>
                      <span className="hidden sm:inline">Comparing...</span>
                      <span className="sm:hidden">...</span>
                    </span>
                  ) : (
                    <span className="flex items-center justify-center gap-1 sm:gap-2">
                      <span>⚖️</span>
                      <span className="hidden sm:inline">Compare Models</span>
                      <span className="sm:hidden">Compare</span>
                    </span>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

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

        {singleResult && (
          <div className="bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 mb-4 sm:mb-6">
            <h2 className="text-lg sm:text-xl md:text-2xl font-semibold text-white mb-3 sm:mb-4">
              📊 Analysis Result - {formatModelName(singleResult.model)}
            </h2>
            
            <div className="grid grid-cols-1 xl:grid-cols-5 gap-4 sm:gap-6 mb-4 sm:mb-6">
              <div className="xl:col-span-3 space-y-3 sm:space-y-4">
                <div className="flex items-center gap-2 sm:gap-4">
                  <ScoreBadge 
                    verdict={singleResult.prediction === "Real" ? "reliable" : "misinformation"} 
                    confidence={singleResult.confidence / 100} 
                  />
                </div>
                
                <div className="p-3 sm:p-4 bg-gray-700 rounded-lg border border-gray-600">
                  <p className="text-xs sm:text-sm font-medium text-gray-300 mb-1">Confidence:</p>
                  <div className="flex items-center">
                    <div className="flex-1 bg-gray-600 rounded-full h-2 sm:h-3 mr-2 sm:mr-3">
                      <div 
                        className={`h-2 sm:h-3 rounded-full ${
                          singleResult.prediction === "Real" 
                            ? (singleResult.confidence > 70 ? "bg-emerald-500" : 
                               singleResult.confidence > 50 ? "bg-amber-500" : "bg-rose-500")
                            : (singleResult.confidence > 70 ? "bg-rose-500" : 
                               singleResult.confidence > 50 ? "bg-amber-500" : "bg-emerald-500")
                        }`}
                        style={{ width: `${singleResult.confidence}%` }}
                      ></div>
                    </div>
                    <span className="text-base sm:text-lg font-bold text-white">
                      {singleResult.confidence.toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">
                    {singleResult.prediction === "Real" 
                      ? "Higher confidence = More likely reliable" 
                      : "Higher confidence = More likely misinformation"}
                  </p>
                </div>

                <RationaleCard 
                  text={claim} 
                  rationales={singleResult.rationales}
                />
              </div>
              
              <div className="xl:col-span-2">
                <PropGraph 
                  data={{
                    probabilities: {
                      misinformation: singleResult.prediction === "Fake" ? singleResult.confidence / 100 : (100 - singleResult.confidence) / 100,
                      reliable: singleResult.prediction === "Real" ? singleResult.confidence / 100 : (100 - singleResult.confidence) / 100
                    },
                    confidence: singleResult.confidence / 100
                  }}
                  type="pie"
                />
              </div>
            </div>
            
            {singleResult.rationales && singleResult.rationales.length > 0 && (
              <div className="mt-4 sm:mt-6 p-3 sm:p-4 bg-blue-900/30 border border-blue-700/50 rounded-lg">
                <p className="text-xs sm:text-sm font-medium text-blue-300 mb-2">
                  🎯 Argument Rationales 
                  <span title="Words most important for the model's decision are highlighted.">
                    <InformationCircleIcon className="h-3 w-3 sm:h-4 sm:w-4 inline text-blue-400 ml-1" />
                  </span>
                </p>
                <p className="text-xs sm:text-sm text-blue-200">
                  This model identified key argumentative elements in the text to make its prediction.
                  Higher rationale scores indicate more important words for the decision.
                </p>
              </div>
            )}
          </div>
        )}

        {compareResult && (
          <div className="bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 mb-4 sm:mb-6">
            <h2 className="text-lg sm:text-xl md:text-2xl font-semibold text-white mb-4 sm:mb-6">
              ⚖️ Model Comparison Results
            </h2>
            
            <div className="mb-4 sm:mb-6 p-3 sm:p-4 bg-gray-700 border border-gray-600 rounded-lg">
              <p className="text-xs sm:text-sm font-medium text-gray-300 mb-1">Analyzed Claim:</p>
              <p className="text-sm sm:text-base text-gray-100 italic break-words">"{compareResult.input}"</p>
            </div>

            <div className="mb-6 sm:mb-8">
              <h3 className="text-base sm:text-lg font-semibold text-gray-200 mb-3 sm:mb-4">Confidence Comparison</h3>
              <div className="h-64 sm:h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={getChartData()} margin={{ top: 20, right: 30, left: 60, bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="model" 
                      tick={{ fontSize: 10 }}
                      angle={-45}
                      textAnchor="end"
                      height={80}
                      label={{ value: 'Models', position: 'insideBottom', offset: -15 }}
                    />
                    <YAxis 
                      domain={[0, 100]}
                      tick={{ fontSize: 10 }}
                      label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }}
                    />
                    <Tooltip 
                      formatter={(value) => [`${value}%`, 'Confidence']}
                      labelFormatter={(label) => `Model: ${label}`}
                    />
                    <Bar dataKey="confidence">
                      {getChartData().map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.prediction === "Real" ? "#22C55E" : "#EF4444"} 
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4 mb-4 sm:mb-6">
              {compareResult.results.map((result, index) => (
                <div key={result.model} className="p-3 sm:p-4 bg-gray-700 rounded-lg border border-gray-600">
                  <h4 className="text-sm sm:text-base font-semibold text-gray-200 mb-2">{formatModelName(result.model)}</h4>
                  <div className="mb-2 sm:mb-3">
                    <ScoreBadge 
                      verdict={result.prediction === "Real" ? "reliable" : "misinformation"} 
                      confidence={result.confidence / 100} 
                    />
                  </div>
                  <div className="text-xs sm:text-sm text-gray-300">
                    {result.model === "BioBERT" && "Base BioBERT model"}
                    {result.model === "BioBERT_ARG" && "BioBERT + Adaptive Rationale Guidance"}
                    {result.model === "BioBERT_ARG_GNN" && "BioBERT + ARG + Graph Neural Networks"}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6">
          <h2 className="text-lg sm:text-xl md:text-2xl font-semibold text-indigo-100 mb-3 sm:mb-4">🧠 About the Models</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6">
            <div className="p-3 sm:p-4 border border-gray-600 rounded-lg bg-gray-700">
              <h3 className="text-sm sm:text-base font-semibold text-blue-300 mb-2">BioBERT</h3>
              <p className="text-xs sm:text-sm text-gray-300">
                Base biomedical language model trained on PubMed abstracts and full-text papers.
                Specializes in understanding medical and health-related terminology.
              </p>
            </div>
            <div className="p-3 sm:p-4 border border-gray-600 rounded-lg bg-gray-700">
              <h3 className="text-sm sm:text-base font-semibold text-green-300 mb-2">BioBERT + ARG</h3>
              <p className="text-xs sm:text-sm text-gray-300 mb-2">
                Enhanced with <strong>Adaptive Rationale Guidance (ARG)</strong> - a framework that enhances reasoning by adaptively identifying, weighing, and guiding the use of key argumentative elements in health claims for better decision-making.
              </p>
              <p className="text-xs text-gray-400">ARG = Adaptive Rationale Guidance</p>
            </div>
            <div className="p-3 sm:p-4 border border-gray-600 rounded-lg bg-gray-700">
              <h3 className="text-sm sm:text-base font-semibold text-purple-300 mb-2">BioBERT + ARG + GNN</h3>
              <p className="text-xs sm:text-sm text-gray-300 mb-2">
                Most advanced model combining BioBERT, Adaptive Rationale Guidance, and Graph Neural Networks (GNN)
                to understand complex relationships between concepts in medical claims.
              </p>
              <p className="text-xs text-gray-400">ARG = Adaptive Rationale Guidance | GNN = Graph Neural Networks</p>
            </div>
          </div>
        </div>
      </div>
      
      <footer className="mt-8 sm:mt-12 py-4 sm:py-6 text-center text-gray-400 text-xs sm:text-sm border-t border-gray-800">
        <span className="block sm:inline">© {new Date().getFullYear()} Health Misinfo Detector. Built with React, Tailwind CSS, and FastAPI.</span>
        <span className="block sm:inline sm:ml-2">| <a href="https://github.com/JyothikaGolla/Health-MisInformation-Detector" className="underline hover:text-indigo-300">GitHub</a></span>
      </footer>
    </div>
  );
};

export default App;