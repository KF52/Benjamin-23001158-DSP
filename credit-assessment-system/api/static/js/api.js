// static/js/api.js
async function fetchWithToken(url, options = {}) {
    // Include cookies in the request
    const response = await fetch(url, {
        ...options,
        credentials: 'same-origin',
    });
    
    // Check if token needs to be refreshed
    if (response.status === 401) {
        // Try to refresh the token
        const refreshResponse = await fetch('/refresh-token/', {
            method: 'POST',
            credentials: 'same-origin',
        });
        
        if (refreshResponse.ok) {
            // Token refreshed, retry the original request
            return fetch(url, {
                ...options,
                credentials: 'same-origin',
            });
        } else {
            // Refresh failed, redirect to login
            window.location.href = '/login/';
        }
    }
    
    return response;
}