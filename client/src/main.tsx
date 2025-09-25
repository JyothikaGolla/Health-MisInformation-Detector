import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

const root = document.getElementById('root');
if (root) {
  root.setAttribute('lang', 'en');
  root.setAttribute('aria-label', 'Health Misinformation Detector Application');
}

ReactDOM.createRoot(root!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
