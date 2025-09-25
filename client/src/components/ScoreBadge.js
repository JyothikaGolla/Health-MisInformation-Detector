import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function ScoreBadge({ verdict, confidence }) {
    const color = verdict === 'true' ? 'green' : verdict === 'fake' ? 'red' : 'gray';
    return (_jsxs("div", { className: `inline-flex items-center gap-2 border rounded px-3 py-1`, children: [_jsx("span", { className: `w-2 h-2 rounded-full`, style: { background: color } }), _jsx("span", { className: "font-semibold capitalize", children: verdict }), _jsxs("span", { className: "opacity-70 text-sm", children: ["(", (confidence * 100).toFixed(0), "%)"] })] }));
}
