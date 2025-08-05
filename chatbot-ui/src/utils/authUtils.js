/**
 * Authentication utility functions for the application
 */

/**
 * Gets the current user's email from localStorage
 * @returns {string} The user's email or a default value if not found
 */
export const getUserEmail = () => {
  // First check the localStorage/sessionStorage keys used by authUtils
  const email = localStorage.getItem('userEmail') || sessionStorage.getItem('userEmail');
  
  // If not found, check the App.jsx's storage keys (loggedInEmail)
  const appEmail = localStorage.getItem('loggedInEmail');
  
  if (email) {
    return email;
  } else if (appEmail) {
    // Store in userEmail key for future use
    localStorage.setItem('userEmail', appEmail);
    return appEmail;
  }
  
  // Return a fallback for development only
  return 'guest@example.com';
};

/**
 * Gets authentication headers for API requests
 * @returns {Object} Headers object with auth information
 */
export const getAuthHeaders = () => {
  return {
    'Content-Type': 'application/json',
    'X-User-Email': getUserEmail()
  };
};

/**
 * Gets the base API URL for backend calls
 * @returns {string} The base API URL
 */
export const getBaseApiUrl = () => {
  // Use the backend server URL instead of the frontend URL
  return 'http://localhost:8000';
};
