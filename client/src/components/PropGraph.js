import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
export default function PropGraph({ cues }) {
    const data = [
        { name: 'Shares', value: cues?.shares ?? 0 },
        { name: 'Influencers', value: cues?.influencers ?? 0 }
    ];
    return (_jsx("div", { className: "border rounded p-3", children: _jsx("div", { className: "h-60", children: _jsx(ResponsiveContainer, { children: _jsxs(BarChart, { data: data, children: [_jsx(XAxis, { dataKey: "name" }), _jsx(YAxis, { allowDecimals: false }), _jsx(Tooltip, {}), _jsx(Bar, { dataKey: "value" })] }) }) }) }));
}
