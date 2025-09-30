// API Types for Health Misinformation Detector

export interface PredictionRequest {
  text: string;
  model_name: string;
}

export interface PredictionResult {
  prediction: number;
  confidence: number;
  label: "reliable" | "misinformation";
  probabilities: {
    misinformation: number;
    reliable: number;
  };
  rationales?: number[][];
  model_used: string;
}

export interface ModelResult {
  model: string;
  prediction: "Real" | "Fake";
  confidence: number;
  label: string;
  rationales?: number[];
}

export interface CompareResult {
  input: string;
  results: ModelResult[];
}

export interface ApiError {
  error: string;
}

// Health endpoint response
export interface HealthResponse {
  status: string;
  models_loaded: number;
  device: string;
}

// Root endpoint response  
export interface RootResponse {
  message: string;
  description: string;
  version: string;
  endpoints: {
    predict: string;
    health: string;
    docs: string;
    openapi: string;
  };
  models_available: string[];
  status: string;
}