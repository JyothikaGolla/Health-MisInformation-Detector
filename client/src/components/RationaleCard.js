import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function RationaleCard({ text, rationales }) {
    // If rationales are provided, we can highlight important words
    // For now, we'll just display the text and mention rationale availability
    return (_jsxs("div", { className: "border rounded-lg p-3 sm:p-4 shadow-sm bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700", children: [_jsx("div", { className: "mb-2", children: _jsx("h3", { className: "text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300", children: "Analysis Text" }) }), _jsx("p", { className: "text-sm sm:text-base text-gray-900 dark:text-gray-100 leading-relaxed break-words", children: text }), rationales && rationales.length > 0 && (_jsx("div", { className: "mt-2 sm:mt-3 p-2 sm:p-3 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800", children: _jsxs("p", { className: "text-xs text-blue-700 dark:text-blue-300", children: ["\u2728 This model identified ", rationales.length, " rationale scores for different text elements. Higher scores indicate more important words for the prediction."] }) }))] }));
}
