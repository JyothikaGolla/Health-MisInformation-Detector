// API Configuration
const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE_URL = isDevelopment
    ? 'http://127.0.0.1:8000'
    : 'https://your-backend-app.onrender.com'; // Replace with your actual Render URL
export { API_BASE_URL };
