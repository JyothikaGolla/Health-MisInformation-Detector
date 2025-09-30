import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell } from 'recharts';
export default function PropGraph({ data, type = 'pie' }) {
    if (!data?.probabilities) {
        return (_jsx("div", { className: "border rounded p-3 bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700", children: _jsx("p", { className: "text-gray-500 dark:text-gray-400 text-center", children: "No probability data available" }) }));
    }
    const chartData = [
        {
            name: 'Misinformation',
            value: Math.round(data.probabilities.misinformation * 100),
            fullValue: data.probabilities.misinformation
        },
        {
            name: 'Reliable',
            value: Math.round(data.probabilities.reliable * 100),
            fullValue: data.probabilities.reliable
        }
    ];
    const colors = ['#EF4444', '#22C55E']; // Brighter red for misinformation, Brighter green for reliable
    const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, name, value }) => {
        const RADIAN = Math.PI / 180;
        const radius = innerRadius + (outerRadius - innerRadius) * 1.4;
        const x = cx + radius * Math.cos(-midAngle * RADIAN);
        const y = cy + radius * Math.sin(-midAngle * RADIAN);
        // Adjust positioning for better visibility on mobile
        let adjustedX = x;
        if (x < cx && name === 'Misinformation') {
            // For misinformation label on the left, move it further right
            adjustedX = Math.max(x, 15);
        }
        return (_jsxs("text", { x: adjustedX, y: y, fill: "#E5E7EB", textAnchor: adjustedX > cx ? 'start' : 'end', dominantBaseline: "central", fontSize: 9, fontWeight: "500", className: "text-xs sm:text-sm", children: [_jsx("tspan", { className: "hidden sm:inline", children: `${name}: ${value}%` }), _jsx("tspan", { className: "sm:hidden", children: `${value}%` })] }));
    };
    if (type === 'pie') {
        return (_jsxs("div", { className: "border rounded-lg p-3 sm:p-4 md:p-6 lg:p-3 bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700", children: [_jsx("h3", { className: "text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 sm:mb-3 text-center", children: "Probability Distribution" }), _jsx("div", { className: "px-2 sm:px-3 md:px-3 lg:px-7 h-40 sm:h-44 md:h-48", style: { width: '100%' }, children: _jsx(ResponsiveContainer, { width: "100%", height: "100%", children: _jsxs(PieChart, { margin: { top: 10, right: 30, bottom: 10, left: 30 }, children: [_jsx(Pie, { data: chartData, cx: "50%", cy: "50%", innerRadius: 15, outerRadius: 30, dataKey: "value", label: renderCustomLabel, labelLine: false, children: chartData.map((entry, index) => (_jsx(Cell, { fill: colors[index], stroke: "#374151", strokeWidth: 1 }, `cell-${index}`))) }), _jsx(Tooltip, { formatter: (value) => `${value}%`, contentStyle: {
                                        fontSize: '12px',
                                        backgroundColor: '#374151',
                                        border: '1px solid #6B7280',
                                        borderRadius: '6px',
                                        color: '#F3F4F6'
                                    } })] }) }) }), _jsxs("div", { className: "flex justify-center gap-3 sm:gap-4 mt-2 sm:mt-3 text-xs sm:text-sm", children: [_jsxs("div", { className: "flex items-center gap-1 sm:gap-2", children: [_jsx("div", { className: "w-2 h-2 sm:w-3 sm:h-3 rounded-full", style: { backgroundColor: colors[0] } }), _jsxs("span", { className: "text-gray-700 dark:text-gray-300", children: [_jsx("span", { className: "hidden sm:inline", children: "Misinformation: " }), _jsx("span", { className: "sm:hidden", children: "Misinfo: " }), chartData[0].value, "%"] })] }), _jsxs("div", { className: "flex items-center gap-1 sm:gap-2", children: [_jsx("div", { className: "w-2 h-2 sm:w-3 sm:h-3 rounded-full", style: { backgroundColor: colors[1] } }), _jsxs("span", { className: "text-gray-700 dark:text-gray-300", children: [_jsx("span", { className: "hidden sm:inline", children: "Reliable: " }), _jsx("span", { className: "sm:hidden", children: "Reliable: " }), chartData[1].value, "%"] })] })] })] }));
    }
    return (_jsxs("div", { className: "border rounded-lg p-3 sm:p-4 bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700", children: [_jsx("h3", { className: "text-xs sm:text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 sm:mb-3", children: "Model Confidence" }), _jsx("div", { className: "h-48 sm:h-60", children: _jsx(ResponsiveContainer, { children: _jsxs(BarChart, { data: chartData, margin: { top: 10, right: 10, left: 10, bottom: 20 }, children: [_jsx(XAxis, { dataKey: "name", tick: { fontSize: 10 }, className: "text-xs sm:text-sm" }), _jsx(YAxis, { domain: [0, 100], tick: { fontSize: 10 }, className: "text-xs sm:text-sm" }), _jsx(Tooltip, { formatter: (value) => `${value}%`, contentStyle: {
                                    fontSize: '12px',
                                    backgroundColor: '#374151',
                                    border: '1px solid #6B7280',
                                    borderRadius: '6px',
                                    color: '#F3F4F6'
                                } }), _jsx(Bar, { dataKey: "value", children: chartData.map((entry, index) => (_jsx(Cell, { fill: colors[index] }, `cell-${index}`))) })] }) }) })] }));
}
