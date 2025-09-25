import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
export default function ClaimForm({ onAnalyze, loading }) {
    const [claim, setClaim] = useState('Turmeric cures cancer');
    const [postId, setPostId] = useState('0');
    const submit = (e) => {
        e.preventDefault();
        onAnalyze({ claim, meta: { postId } });
    };
    return (_jsxs("form", { onSubmit: submit, className: "grid gap-3", children: [_jsx("textarea", { className: "border rounded p-3", rows: 4, value: claim, onChange: e => setClaim(e.target.value) }), _jsxs("div", { className: "flex gap-3 items-center", children: [_jsx("label", { className: "text-sm opacity-70", children: "Post ID" }), _jsx("input", { className: "border rounded px-2 py-1 w-24", value: postId, onChange: e => setPostId(e.target.value) }), _jsx("button", { disabled: loading, className: "rounded px-4 py-2 border", children: loading ? 'Analyzing…' : 'Analyze' })] })] }));
}
