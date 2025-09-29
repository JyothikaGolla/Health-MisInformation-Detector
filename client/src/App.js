import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
// App.tsx
import { useState } from "react";
import { InformationCircleIcon } from '@heroicons/react/24/outline';
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
const App = () => {
    const [claim, setClaim] = useState("");
    const [analyzeModel, setAnalyzeModel] = useState("BioBERT");
    const [singleResult, setSingleResult] = useState(null);
    const [compareResult, setCompareResult] = useState(null);
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
    // Helper function to format model names for display
    const formatModelName = (modelName) => {
        return modelName.replace(/_/g, "+");
    };
    const handleAnalyze = async () => {
        if (!claim.trim())
            return;
        setLoading(true);
        setError("");
        setSingleResult(null);
        setCompareResult(null);
        try {
            const response = await axios.post("http://127.0.0.1:8000/predict", {
                text: claim,
                model_type: analyzeModel,
            });
            const result = {
                model: analyzeModel,
                prediction: response.data.pred === 0 ? "Fake" : "Real",
                confidence: Math.max(...response.data.probs[0]) * 100,
                rationales: response.data.rationales?.[0]
            };
            setSingleResult(result);
        }
        catch (err) {
            setError(err.response?.data?.error || "Failed to analyze claim");
        }
        finally {
            setLoading(false);
        }
    };
    const handleCompare = async () => {
        if (!claim.trim())
            return;
        setLoading(true);
        setError("");
        setSingleResult(null);
        setCompareResult(null);
        try {
            const promises = models.map(model => axios.post("http://127.0.0.1:8000/predict", {
                text: claim,
                model_type: model,
            }));
            const responses = await Promise.all(promises);
            const results = responses.map((response, index) => ({
                model: models[index],
                prediction: response.data.pred === 0 ? "Fake" : "Real",
                confidence: Math.max(...response.data.probs[0]) * 100,
                rationales: response.data.rationales?.[0]
            }));
            setCompareResult({
                input: claim,
                results
            });
        }
        catch (err) {
            setError(err.response?.data?.error || "Failed to compare models");
        }
        finally {
            setLoading(false);
        }
    };
    const handleSampleClaim = (sampleClaim) => {
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
    return (_jsxs("div", { className: "dark bg-gradient-to-br from-gray-900 to-gray-800 min-h-screen", children: [_jsx("header", { className: "sticky top-0 z-20 bg-gray-900/80 backdrop-blur border-b border-gray-800 shadow-sm flex items-center justify-between px-6 py-3", children: _jsx("div", { className: "flex items-center gap-3", children: _jsx("span", { className: "text-2xl font-bold text-indigo-300", children: "\uD83D\uDD0D Health Misinfo Detector" }) }) }), _jsxs("div", { className: "max-w-6xl mx-auto px-2 sm:px-6", children: [_jsxs("div", { className: "text-center mb-8 py-8", children: [_jsx("h1", { className: "text-4xl font-bold text-indigo-100 mb-6", children: "Analyze health claims using advanced AI models including BioBERT with Adaptive Rationale Guidance and Graph Neural Networks" }), _jsxs("div", { className: "flex flex-col items-center justify-center gap-4", children: [_jsx("div", { className: "flex-shrink-0 flex items-center justify-center", children: _jsx(InformationCircleIcon, { className: "h-16 w-16 md:h-24 md:w-24 text-blue-300" }) }), _jsx("div", { className: "max-w-md text-lg text-gray-300 text-center", children: _jsxs("span", { children: ["Choose a claim and model to get started.", _jsx("br", {}), _jsx("span", { className: "text-sm text-gray-400", children: "Results are AI predictions, not medical advice." })] }) })] })] }), _jsxs("div", { className: "bg-gray-800 rounded-xl shadow-lg p-6 mb-6", children: [_jsx("h2", { className: "text-xl font-semibold text-white mb-4", children: "Try Sample Health Claims:" }), _jsx("div", { className: "flex flex-wrap gap-2", children: sampleClaims.map((sampleClaim, index) => (_jsx("button", { className: "text-left px-3 py-2 bg-gray-700 hover:bg-blue-800 rounded-lg border border-gray-600 hover:border-blue-500 transition-all duration-200 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 text-gray-200", onClick: () => handleSampleClaim(sampleClaim), tabIndex: 0, "aria-label": `Try sample claim: ${sampleClaim}`, children: sampleClaim }, index))) })] }), _jsxs("div", { className: "bg-gray-800 rounded-xl shadow-lg p-6 mb-6", children: [_jsx("h2", { className: "text-xl font-semibold text-white mb-4", children: "Enter Health Claim:" }), _jsxs("div", { className: "flex flex-col sm:flex-row gap-4 items-start sm:items-center mt-2", children: [_jsx("textarea", { className: "w-full p-4 border border-gray-600 bg-gray-700 text-white rounded-lg mb-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none shadow placeholder-gray-300", rows: 4, placeholder: "Enter a health claim to analyze (e.g., 'Vitamin C prevents COVID-19', 'Coffee causes cancer', etc.)...", value: claim, onChange: (e) => setClaim(e.target.value), "aria-label": "Health claim input" }), _jsx("button", { className: "bg-gray-600 text-gray-200 px-4 py-2 rounded-lg font-medium transition-colors duration-200 hover:bg-gray-500 mb-2 shadow", onClick: () => setClaim(""), disabled: !claim.trim(), "aria-label": "Clear claim input", children: "Clear" })] }), _jsxs("div", { className: "flex flex-col sm:flex-row gap-4 items-start sm:items-center", children: [_jsxs("div", { className: "flex items-center gap-2 mt-2", children: [_jsx("label", { htmlFor: "model-select", className: "text-sm font-medium text-gray-200", children: "Model:" }), _jsx("select", { id: "model-select", className: "border border-gray-600 bg-gray-700 text-white rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow", value: analyzeModel, onChange: (e) => setAnalyzeModel(e.target.value), "aria-label": "Model selection", children: models.map((model) => (_jsx("option", { value: model, children: formatModelName(model) }, model))) }), _jsx("span", { className: "ml-1", title: "Select which AI model to use for analysis.", children: _jsx(InformationCircleIcon, { className: "h-4 w-4 text-blue-300" }) })] }), _jsxs("div", { className: "flex gap-3", children: [_jsx("button", { className: "bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow focus:outline-none focus:ring-2 focus:ring-blue-400", onClick: handleAnalyze, disabled: loading || !claim.trim(), "aria-label": "Analyze claim", children: loading ? (_jsxs("span", { className: "flex items-center gap-2", children: [_jsx("span", { className: "animate-spin h-5 w-5 border-b-2 border-white rounded-full" }), "Analyzing..."] })) : "🔍 Analyze" }), _jsx("button", { className: "bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg font-medium transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow focus:outline-none focus:ring-2 focus:ring-green-400", onClick: handleCompare, disabled: loading || !claim.trim(), "aria-label": "Compare models", children: loading ? (_jsxs("span", { className: "flex items-center gap-2", children: [_jsx("span", { className: "animate-spin h-5 w-5 border-b-2 border-white rounded-full" }), "Comparing..."] })) : "⚖️ Compare Models" })] })] })] }), loading && (_jsx("div", { className: "bg-blue-50 dark:bg-gray-800 border-l-4 border-blue-400 dark:border-blue-600 p-4 mb-6 rounded animate-pulse", children: _jsxs("div", { className: "flex items-center", children: [_jsx("div", { className: "animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 dark:border-blue-300 mr-3" }), _jsx("p", { className: "text-blue-700 dark:text-blue-200 font-medium", children: "Processing your health claim..." })] }) })), error && (_jsx("div", { className: "bg-red-50 dark:bg-red-900 border-l-4 border-red-400 dark:border-red-700 p-4 mb-6 rounded animate-shake", children: _jsxs("p", { className: "text-red-700 dark:text-red-200 font-medium", children: ["\u274C Error: ", error] }) })), singleResult && (_jsxs("div", { className: "bg-gray-800 rounded-xl shadow-lg p-6 mb-6", children: [_jsxs("h2", { className: "text-2xl font-semibold text-white mb-4", children: ["\uD83D\uDCCA Analysis Result - ", formatModelName(singleResult.model)] }), _jsxs("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-6", children: [_jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "p-4 bg-gray-700 rounded-lg border border-gray-600", children: [_jsx("p", { className: "text-sm font-medium text-gray-300 mb-1", children: "Prediction:" }), _jsx("p", { className: `text-2xl font-bold ${singleResult.prediction === "Real" ? "text-emerald-400" : "text-rose-400"}`, children: singleResult.prediction === "Real" ? "✅ Real" : "❌ Fake" })] }), _jsxs("div", { className: "p-4 bg-gray-700 rounded-lg border border-gray-600", children: [_jsx("p", { className: "text-sm font-medium text-gray-300 mb-1", children: "Confidence:" }), _jsxs("div", { className: "flex items-center", children: [_jsx("div", { className: "flex-1 bg-gray-600 rounded-full h-3 mr-3", children: _jsx("div", { className: `h-3 rounded-full ${singleResult.confidence > 70 ? "bg-emerald-500" :
                                                                        singleResult.confidence > 50 ? "bg-amber-500" : "bg-rose-500"}`, style: { width: `${singleResult.confidence}%` } }) }), _jsxs("span", { className: "text-lg font-bold text-white", children: [singleResult.confidence.toFixed(1), "%"] })] })] })] }), _jsxs("div", { className: "p-4 bg-gray-700 rounded-lg border border-gray-600", children: [_jsx("p", { className: "text-sm font-medium text-gray-300 mb-2", children: "Input Claim:" }), _jsxs("p", { className: "text-gray-100 italic", children: ["\"", claim, "\""] })] })] }), singleResult.rationales && singleResult.rationales.length > 0 && (_jsxs("div", { className: "mt-6 p-4 bg-blue-900/30 border border-blue-700/50 rounded-lg", children: [_jsxs("p", { className: "text-sm font-medium text-blue-300 mb-2", children: ["\uD83C\uDFAF Argument Rationales ", _jsx("span", { title: "Words most important for the model's decision are highlighted.", children: _jsx(InformationCircleIcon, { className: "h-4 w-4 inline text-blue-400" }) })] }), _jsx("p", { className: "text-sm text-blue-200", children: "This model identified key argumentative elements in the text to make its prediction. Higher rationale scores indicate more important words for the decision." })] }))] })), compareResult && (_jsxs("div", { className: "bg-gray-800 rounded-xl shadow-lg p-6 mb-6", children: [_jsx("h2", { className: "text-2xl font-semibold text-white mb-6", children: "\u2696\uFE0F Model Comparison Results" }), _jsxs("div", { className: "mb-6 p-4 bg-gray-700 border border-gray-600 rounded-lg", children: [_jsx("p", { className: "text-sm font-medium text-gray-300 mb-1", children: "Analyzed Claim:" }), _jsxs("p", { className: "text-gray-100 italic", children: ["\"", compareResult.input, "\""] })] }), _jsxs("div", { className: "mb-8", children: [_jsx("h3", { className: "text-lg font-semibold text-gray-200 mb-4", children: "Confidence Comparison" }), _jsx("div", { className: "h-80", children: _jsx(ResponsiveContainer, { width: "100%", height: "100%", children: _jsxs(BarChart, { data: getChartData(), margin: { top: 20, right: 30, left: 60, bottom: 80 }, children: [_jsx(CartesianGrid, { strokeDasharray: "3 3" }), _jsx(XAxis, { dataKey: "model", tick: { fontSize: 12 }, angle: -45, textAnchor: "end", height: 80, label: { value: 'Models', position: 'insideBottom', offset: -15 } }), _jsx(YAxis, { domain: [0, 100], tick: { fontSize: 12 }, label: { value: 'Confidence (%)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } } }), _jsx(Tooltip, { formatter: (value) => [`${value}%`, 'Confidence'], labelFormatter: (label) => `Model: ${label}` }), _jsx(Bar, { dataKey: "confidence", children: getChartData().map((entry, index) => (_jsx(Cell, { fill: entry.prediction === "Real" ? "#10B981" : "#F87171" }, `cell-${index}`))) })] }) }) })] }), _jsx("div", { className: "overflow-x-auto", children: _jsxs("table", { className: "w-full border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden", children: [_jsx("thead", { className: "bg-gray-100 dark:bg-gray-800", children: _jsxs("tr", { children: [_jsx("th", { className: "px-6 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-200", children: "Model" }), _jsx("th", { className: "px-6 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-200", children: "Prediction" }), _jsx("th", { className: "px-6 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-200", children: "Confidence" }), _jsx("th", { className: "px-6 py-3 text-left text-sm font-semibold text-gray-700 dark:text-gray-200", children: "Features" })] }) }), _jsx("tbody", { className: "divide-y divide-gray-200 dark:divide-gray-700", children: compareResult.results.map((result, index) => (_jsxs("tr", { className: index % 2 === 0 ? "bg-white dark:bg-gray-900" : "bg-gray-50 dark:bg-gray-800", children: [_jsx("td", { className: "px-6 py-4 text-sm font-medium text-gray-900 dark:text-gray-100", children: formatModelName(result.model) }), _jsx("td", { className: "px-6 py-4", children: _jsx("span", { className: `inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${result.prediction === "Real"
                                                                ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                                                                : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"}`, children: result.prediction === "Real" ? "✅ Real" : "❌ Fake" }) }), _jsx("td", { className: "px-6 py-4 text-sm text-gray-700 dark:text-gray-200", children: _jsxs("div", { className: "flex items-center", children: [_jsx("div", { className: "flex-1 bg-gray-200 rounded-full h-2 mr-3 w-20", children: _jsx("div", { className: `h-2 rounded-full ${result.confidence > 70 ? "bg-green-500" :
                                                                            result.confidence > 50 ? "bg-yellow-500" : "bg-red-500"}`, style: { width: `${result.confidence}%` } }) }), _jsxs("span", { className: "font-medium", children: [result.confidence.toFixed(1), "%"] })] }) }), _jsxs("td", { className: "px-6 py-4 text-sm text-gray-700 dark:text-gray-200", children: [result.model === "BioBERT" && "Base BioBERT model", result.model === "BioBERT_ARG" && "BioBERT + Adaptive Rationale Guidance", result.model === "BioBERT_ARG_GNN" && "BioBERT + Adaptive Rationale Guidance + Graph Neural Networks"] })] }, result.model))) })] }) })] })), _jsxs("div", { className: "bg-gray-800 rounded-xl shadow-lg p-6", children: [_jsx("h2", { className: "text-2xl font-semibold text-indigo-100 mb-4", children: "\uD83E\uDDE0 About the Models" }), _jsxs("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-6", children: [_jsxs("div", { className: "p-4 border border-gray-600 rounded-lg bg-gray-700", children: [_jsx("h3", { className: "font-semibold text-blue-300 mb-2", children: "BioBERT" }), _jsx("p", { className: "text-sm text-gray-300", children: "Base biomedical language model trained on PubMed abstracts and full-text papers. Specializes in understanding medical and health-related terminology." })] }), _jsxs("div", { className: "p-4 border border-gray-600 rounded-lg bg-gray-700", children: [_jsx("h3", { className: "font-semibold text-green-300 mb-2", children: "BioBERT + ARG" }), _jsxs("p", { className: "text-sm text-gray-300 mb-2", children: ["Enhanced with ", _jsx("strong", { children: "Adaptive Rationale Guidance (ARG)" }), " - argument mining capabilities to identify and weigh important argumentative elements in health claims for better reasoning."] }), _jsx("p", { className: "text-xs text-gray-400", children: "ARG = Adaptive Rationale Guidance" })] }), _jsxs("div", { className: "p-4 border border-gray-600 rounded-lg bg-gray-700", children: [_jsx("h3", { className: "font-semibold text-purple-300 mb-2", children: "BioBERT + ARG + GNN" }), _jsx("p", { className: "text-sm text-gray-300 mb-2", children: "Most advanced model combining BioBERT, Adaptive Rationale Guidance, and Graph Neural Networks (GNN) to understand complex relationships between concepts in medical claims." }), _jsx("p", { className: "text-xs text-gray-400", children: "ARG = Adaptive Rationale Guidance | GNN = Graph Neural Networks" })] })] })] })] }), _jsx("footer", { className: "mt-12 py-6 text-center text-gray-400 text-sm border-t border-gray-800", children: _jsxs("span", { children: ["\u00A9 ", new Date().getFullYear(), " Health Misinfo Detector. Built with React, Tailwind CSS, and FastAPI. | ", _jsx("a", { href: "https://github.com/JyothikaGolla/Health-MisInformation-Detector", className: "underline hover:text-indigo-300", children: "GitHub" })] }) })] }));
};
export default App;
