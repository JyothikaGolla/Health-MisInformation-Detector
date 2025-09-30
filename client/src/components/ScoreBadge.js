import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function ScoreBadge({ verdict, confidence }) {
    // Map backend labels to colors - backend returns "reliable" or "misinformation"
    const getColor = (label) => {
        switch (label.toLowerCase()) {
            case 'reliable':
                return '#22C55E'; // Bright green
            case 'misinformation':
                return '#EF4444'; // Bright red
            case 'real':
                return '#22C55E'; // Bright green
            case 'fake':
                return '#EF4444'; // Bright red
            default:
                return '#6B7280'; // Gray
        }
    };
    const color = getColor(verdict);
    return (_jsxs("div", { className: `inline-flex items-center gap-1 sm:gap-2 border rounded px-2 sm:px-3 py-1 bg-gray-50 dark:bg-gray-800`, children: [_jsx("span", { className: `w-2 h-2 rounded-full`, style: { background: color } }), _jsx("span", { className: "text-sm sm:text-base font-semibold capitalize", children: verdict }), _jsxs("span", { className: "opacity-70 text-xs sm:text-sm", children: ["(", Math.round(confidence * 100), "%)"] })] }));
}
