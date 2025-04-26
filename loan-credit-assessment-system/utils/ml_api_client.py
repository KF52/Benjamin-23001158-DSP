import requests, json
from django.conf import settings

class MLApiClient:
    """Client class for interacting with the ML API service"""
    
    def __init__(self, base_url=None):
        self.base_url = base_url or settings.FASTAPI_BASE_URL
    
    def predict(self, data, request=None):
        """Send a prediction request to the FastAPI service"""
        url = f"{self.base_url}/predict"

        ## Initialize the Content Type 
        headers = {"Content-Type": "application/json"}

        ## Add authentication token if request is provided 
        if request and request.COOKIES.get("access_token"):
            headers['Authorization'] = f"Bearer {request.COOKIES.get('access_token')}"
        try:
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=10  # Add timeout for safety
            )
            response.raise_for_status()
            
            # Get response data
            response_data = response.json()
            
            # Print the response for debugging
            print("FastAPI response:", response_data)

            # Parse the confidence value as float (removing % if present)
            if 'confidence' in response_data:
                confidence_str = response_data['confidence']
                if isinstance(confidence_str, str):
                    confidence_str = confidence_str.strip('%')
                response_data['confidence'] = float(confidence_str)

            # Make sure explanation data is properly formatted if it exists
            if 'explanation' in response_data and response_data['explanation']:
                # The explanation data is already parsed JSON from the FastAPI response
                print("Explanation data received")
            
            return response_data
        except requests.exceptions.RequestException as e:
            print(f"Error calling ML API: {str(e)}")
            if hasattr(e, 'response') and e.response.status_code == 401:
                raise Exception("Authentication to ML Service has Failed Please Login again")
            raise Exception(f"Failed to communicate with ML service: {str(e)}")

    def health(self):
        """Health check endpoint in the FastAPI Server"""
        url = f"{self.base_url}/health"
        try:
            response = requests.get(url, timeout=5)  # Add timeout
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"ML API health check failed: {str(e)}")
            return False

    def reload_models(self, request=None):
        """Tell the FastAPI service to reload the active model"""
        url = f"{self.base_url}/reload-models"
        
        headers = {"Content-Type": "application/json"}
        
        if request and request.COOKIES.get("access_token"):
            headers['Authorization'] = f"Bearer {request.COOKIES.get('access_token')}"
        
        try:
            response = requests.post(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error reloading FastAPI models: {str(e)}")
            return {"success": False, "error": str(e)}

# For backward compatibility, keeping the standalone functions
def predict(data, request = None):
    """Legacy function that uses the MLApiClient class"""
    client = MLApiClient()
    return client.predict(data, request)

def health():
    """Legacy function that uses the MLApiClient class"""
    client = MLApiClient()
    return client.health()

def reload_models(request=None):
    """Legacy function that uses the MLApiClient class"""
    client = MLApiClient()
    return client.reload_models(request)