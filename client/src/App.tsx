import React, { useState } from "react";

function App() {
  const [claim, setClaim] = useState("");
  const [model, setModel] = useState("biobert");
  const [result, setResult] = useState<any>(null);
  const [compareResult, setCompareResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const onAnalyze = async () => {
    setLoading(true);
    setResult(null);
    setCompareResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim, model_name: model }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Analyze failed:", error);
      alert("Error analyzing claim. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  const onCompare = async () => {
    setLoading(true);
    setResult(null);
    setCompareResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setCompareResult(data);
    } catch (error) {
      console.error("Compare failed:", error);
      alert("Error comparing models. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  const getPredictionColor = (prediction: string) => {
    if (prediction.toLowerCase().includes("fake")) return "#dc3545"; // red
    if (prediction.toLowerCase().includes("true")) return "#28a745"; // green
    return "#333";
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "Arial, sans-serif" }}>
      <h1>Health Misinformation Detector</h1>

      <textarea
        rows={3}
        value={claim}
        onChange={(e) => setClaim(e.target.value)}
        placeholder="Enter a claim..."
        style={{ width: "100%", padding: "10px", fontSize: "16px" }}
      />
      <br />

      <label style={{ marginTop: "10px", display: "block" }}>
        Select Model:
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          style={{ marginLeft: "10px", padding: "5px", fontSize: "14px" }}
        >
          <option value="biobert">BioBERT</option>
          <option value="minilm">MiniLM</option>
          <option value="biobert+arg">BioBERT+ARG</option>
        </select>
      </label>

      <div style={{ marginTop: "10px" }}>
        <button
          onClick={onAnalyze}
          disabled={loading || !claim}
          style={{
            marginRight: "10px",
            padding: "10px 20px",
            fontSize: "16px",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: loading || !claim ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        <button
          onClick={onCompare}
          disabled={loading || !claim}
          style={{
            padding: "10px 20px",
            fontSize: "16px",
            backgroundColor: "#28a745",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: loading || !claim ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Comparing..." : "Compare Models"}
        </button>
      </div>

      {result && (
        <div
          style={{
            marginTop: "2rem",
            padding: "1rem",
            border: "1px solid #ddd",
            borderRadius: "8px",
            backgroundColor: "#f9f9f9",
          }}
        >
          <h3>Result (Single Model)</h3>
          <p><b>Claim:</b> {result.claim}</p>
          <p><b>Model Used:</b> {result.model_used}</p>
          <p>
            <b>Prediction:</b>{" "}
            <span style={{ color: getPredictionColor(result.prediction), fontWeight: "bold" }}>
              {result.prediction}
            </span>
          </p>
          <p><b>Confidence:</b> {result.confidence.toFixed(4)}</p>
        </div>
      )}

      {compareResult && (
        <div
          style={{
            marginTop: "2rem",
            padding: "1rem",
            border: "1px solid #ddd",
            borderRadius: "8px",
            backgroundColor: "#f1f1f1",
          }}
        >
          <h3>Comparison Result</h3>
          <p><b>Claim:</b> {compareResult.claim}</p>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              marginTop: "10px",
              textAlign: "center",
              fontSize: "15px",
            }}
          >
            <thead>
              <tr style={{ background: "#007bff", color: "white" }}>
                <th style={{ border: "1px solid #ccc", padding: "10px" }}>Model</th>
                <th style={{ border: "1px solid #ccc", padding: "10px" }}>Prediction</th>
                <th style={{ border: "1px solid #ccc", padding: "10px" }}>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(compareResult.results).map(([model, res]: any, idx) => (
                <tr
                  key={model}
                  style={{
                    background: idx % 2 === 0 ? "#ffffff" : "#f8f9fa",
                  }}
                >
                  <td style={{ border: "1px solid #ccc", padding: "10px", fontWeight: "bold" }}>
                    {model.toUpperCase()}
                  </td>
                  <td
                    style={{
                      border: "1px solid #ccc",
                      padding: "10px",
                      color: getPredictionColor(res.prediction),
                      fontWeight: "bold",
                    }}
                  >
                    {res.prediction}
                  </td>
                  <td style={{ border: "1px solid #ccc", padding: "10px" }}>
                    {(res.confidence * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
