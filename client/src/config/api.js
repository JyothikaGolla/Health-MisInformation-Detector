// API Configuration with environment detection
const isDevelopment = window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1' ||
    window.location.hostname.includes('localhost');
const isGitHubPages = window.location.hostname.includes('github.io');
// Environment-based API URL selection
const getApiBaseUrl = () => {
    if (isDevelopment) {
        // Local development - try different ports if needed
        return 'http://127.0.0.1:8000';
    }
    else {
        // Production deployment (GitHub Pages, Netlify, etc.)
        return 'https://health-misinformation-detector-1.onrender.com'; // Update with your actual Render URL
    }
};
export const API_BASE_URL = getApiBaseUrl();
// Debug logging (only in development)
if (isDevelopment) {
    console.log('🔧 API Configuration:', {
        environment: 'development',
        apiUrl: API_BASE_URL,
        hostname: window.location.hostname
    });
}
else {
    console.log('🚀 API Configuration:', {
        environment: 'production',
        apiUrl: API_BASE_URL,
        hostname: window.location.hostname
    });
}
