// API Configuration with environment detection
const isDevelopment = 
  window.location.hostname === 'localhost' || 
  window.location.hostname === '127.0.0.1' ||
  window.location.hostname.includes('localhost');

const isCodespace = window.location.hostname.includes('.app.github.dev');

// Environment-based API URL selection
const getApiBaseUrl = (): string => {
  if (isDevelopment) {
    // Local development - try different ports if needed
    return 'http://127.0.0.1:8000';
  } else if (isCodespace) {
    // GitHub Codespace - dynamically construct backend URL
    const hostname = window.location.hostname;
    const backendUrl = hostname.replace(/(-\d+)\.app\.github\.dev$/, '-8000.app.github.dev');
    return `https://${backendUrl}`;
  } else {
    // Fallback for other deployments
    return 'https://literate-garbanzo-x6696pjwjp53995v-8000.app.github.dev';
  }
};

export const API_BASE_URL = getApiBaseUrl();

// Debug logging for troubleshooting
console.log('🔧 API Configuration:', {
  hostname: window.location.hostname,
  isCodespace: window.location.hostname.includes('.app.github.dev'),
  isDevelopment: isDevelopment,
  apiUrl: API_BASE_URL
});