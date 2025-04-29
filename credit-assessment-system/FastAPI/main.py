import pandas as pd
import numpy as np
import os
import json
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, Depends 
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Union, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.neighbors import NearestNeighbors
import time
from datetime import datetime
import threading

## XAI Service
import fix_warnings  # Fix warning issues for SHAP
from xai_service import ShapExplainer

## Authentication Utility
from auth import get_current_user

# Create FastAPI app
app = FastAPI(
    title="Credit Score Prediction API",
    description="API for predicting credit scores based on applicants' financial and personal details",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

## Add CORS Middleware to allow responses from Django
app.add_middleware(
    CORSMiddleware,
    ## Allow communication with the Django App 
    allow_origins=["http://127.0.0.1:8000"], 
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model management
class ModelManager:
    def __init__(self):
        self.active_model = None
        self.active_model_path = "active_model.pkl"
        self.active_model_metadata_path = "active_model_metadata.json"
        self.last_modified_time = 0
        self.last_file_size = 0
        self.requires_scaling = False  # Default to not scaling
        self.model_lock = threading.RLock()  # Reentrant lock for thread safety
        self.load_model()  # Initial load
    
    def load_model(self):
        """Load the active model if it exists and has been updated"""
        try:
            if not os.path.exists(self.active_model_path):
                print(f"Warning: {self.active_model_path} not found")
                return False
                
            # Check if file has been modified
            current_size = os.path.getsize(self.active_model_path)
            current_mtime = os.path.getmtime(self.active_model_path)
            
            if current_size == self.last_file_size and current_mtime <= self.last_modified_time:
                return True
                
            # Force a reload if size or mtime changed
            if current_size != self.last_file_size or current_mtime != self.last_modified_time:
                with self.model_lock:
                    print(f"Loading model from {self.active_model_path} (modified at {datetime.fromtimestamp(current_mtime)})")
                    self.active_model = joblib.load(self.active_model_path)
                    self.last_modified_time = current_mtime
                    self.last_file_size = current_size
                
                # Load model metadata if available
                if os.path.exists(self.active_model_metadata_path):
                    try:
                        with open(self.active_model_metadata_path, 'r') as f:
                            metadata = json.load(f)
                            self.requires_scaling = metadata.get('requires_scaling', False)
                            print(f"Model requires scaling: {self.requires_scaling}")
                    except Exception as e:
                        print(f"Error loading model metadata: {str(e)}")
                        self.requires_scaling = False
                else:
                    print("No model metadata found, assuming scaling not required")
                    self.requires_scaling = False
                    
                print("Active model loaded successfully")
                print(f"Model type: {type(self.active_model).__name__}")
                return True
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def get_model(self):
        """Get the current active model, checking for updates first"""
        # Check if model has been updated
        self.load_model()  
        
        with self.model_lock:
            if self.active_model is None:
                raise ValueError("No active model available")
            return self.active_model
        
        return self.model
    
    def requires_input_scaling(self):
        """Check if the current model requires input scaling"""
        return self.requires_scaling

# Initialize model manager
model_manager = ModelManager()

# Load the serialized preprocessors
standard_scaler = joblib.load('standard_scaler.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')
metadata = joblib.load('preprocessing_metadata.pkl')

# Load training data reference for distance-based confidence calculation
X_train_reference = joblib.load('X_train.pkl')
print(f"Training reference data loaded successfully: {len(X_train_reference)} samples")

# Define request model with proper validation
class PredictionRequest(BaseModel):
    Age: int = Field(..., description="Age of applicant", ge=18)
    Annual_Income: float = Field(..., description="Annual income in dollars")
    Credit_Score: float = Field(..., description="Credit score")
    Employment_Status: str = Field(..., description="Employment status (Employed, Self-Employed, Unemployed)")
    Education_Level: str = Field(..., description="Education level")
    Experience: float = Field(..., description="Years of work experience")
    Loan_Amount: float = Field(..., description="Requested loan amount")
    Loan_Duration: int = Field(..., description="Loan duration in months")
    Marital_Status: str = Field(..., description="Marital status")
    Number_Of_Dependents: int = Field(..., description="Number of dependents")
    Home_Ownership_Status: str = Field(..., description="Home ownership status")
    Monthly_Debt_Payments: float = Field(..., description="Monthly debt payments")
    Credit_Card_Utilization_Rate: float = Field(..., description="Credit card utilization as percentage")
    Number_Of_Open_Credit_Lines: int = Field(..., description="Number of open credit lines")
    Number_Of_Credit_Inquiries: int = Field(..., description="Number of credit inquiries")
    Debt_To_Income_Ratio: float = Field(..., description="Debt to income ratio")
    Bankruptcy_History: str = Field(..., description="Bankruptcy history (Yes/No)")
    Loan_Purpose: str = Field(..., description="Purpose of loan")
    Previous_Loan_Defaults: str = Field(..., description="Previous loan defaults (Yes/No)")
    Payment_History: float = Field(..., description="Payment history score")
    Length_Of_Credit_History: float = Field(..., description="Length of credit history in years")
    Savings_Account_Balance: float = Field(..., description="Savings account balance")
    Checking_Account_Balance: float = Field(..., description="Checking account balance")
    Total_Assets: float = Field(..., description="Total assets value")
    Total_Liabilities: float = Field(..., description="Total liabilities value")
    Monthly_Income: float = Field(..., description="Monthly income")
    Utility_Bills_Payment_History: float = Field(..., description="Utility bills payment history score")
    Job_Tenure: float = Field(..., description="Job tenure in years")
    Net_Worth: float = Field(..., description="Net worth value")
    Total_Debt_To_Income_Ratio: float = Field(..., description="Total debt to income ratio")

    @validator('Bankruptcy_History', 'Previous_Loan_Defaults')
    def validate_binary_fields(cls, v):
        # Accept string values
        if isinstance(v, str):
            if v.lower() in ["yes", "no"]:
                return v
        # Still accept 0/1 for backward compatibility
        if v in [0, 1]:
            return v
        raise ValueError('Value must be either "Yes", "No", 0, or 1')

class PredictionResponse(BaseModel):
    loan_approval: bool = Field(..., description="Predicted Loan approval status (Approved/Rejected)")
    message: str = Field(..., description="Formatted prediction message")
    confidence: str = Field(..., description="Confidence level of the prediction")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")
    explanation: Dict[str, Any] = Field(..., description="SHAP explanation for the prediction")
    
# Preprocessing Function
def preprocess_input(input_data, model_type=None):
    # Convert to dictionary
    data_dict = input_data.copy()

    # Convert Yes/No values to 0/1 for binary fields
    binary_fields = ['Bankruptcy_History', 'Previous_Loan_Defaults']
    for field in binary_fields:
        if field in data_dict:
            if isinstance(data_dict[field], str):
                if data_dict[field].lower() == 'yes':
                    data_dict[field] = 1
                elif data_dict[field].lower() == 'no':
                    data_dict[field] = 0

    # Apply one-hot encoding from saved encoder
    if metadata['onehot_encoding_columns']:
        # Create DataFrame with just the columns needed for one-hot encoding
        onehot_df = pd.DataFrame({col: [data_dict[col]] for col in metadata['onehot_encoding_columns']})
        # Transform using the loaded encoder
        encoded_array = onehot_encoder.transform(onehot_df)
        # Get feature names
        feature_names = onehot_encoder.get_feature_names_out(metadata['onehot_encoding_columns'])
        # Add encoded features to data_dict
        for i, feature in enumerate(feature_names):
            data_dict[feature] = encoded_array[0, i]
        # Remove original categorical columns
        for col in metadata['onehot_encoding_columns']:
            del data_dict[col]

    # Apply Ordinal Encoding from saved encoder
    if metadata['ordinal_encoding_columns']:
        # Create DataFrame with just the columns needed for ordinal encoding
        ordinal_df = pd.DataFrame({col: [data_dict[col]] for col in metadata['ordinal_encoding_columns']})
        # Transform using the loaded encoder
        encoded_array = ordinal_encoder.transform(ordinal_df)
        # Add encoded features to data_dict
        for i, col in enumerate(metadata['ordinal_encoding_columns']):
            data_dict[col] = encoded_array[0, i]

    # LightGBM-specific feature name normalization AFTER one-hot encoding
    if model_type:

        if 'lightgbm' in str(model_type).lower() or 'lgbm' in str(model_type).lower():
            
            # Log all keys in data_dict to see what feature names are actually available
            print("Available feature names:")
            for key in sorted(data_dict.keys()):
                print(f"  - {key}")
            
            # Look for features with partial matches and normalize them
            keys_to_rename = {}
            for key in data_dict.keys():
                if 'Education_Level_High School' in key or ('Education_Level' in key and 'High School' in key):
                    keys_to_rename[key] = 'Education_Level_High_School'
                elif 'Loan_Purpose_Debt Consolidation' in key or ('Loan_Purpose' in key and 'Debt Consolidation' in key):
                    keys_to_rename[key] = 'Loan_Purpose_Debt_Consolidation'
            
            # Apply the renames
            for old_key, new_key in keys_to_rename.items():
                data_dict[new_key] = data_dict[old_key]
                print(f"Remapped feature: {old_key} → {new_key}")
                del data_dict[old_key]
    
    return data_dict

# Feature Scaling Function
def scale_features(data_dict):
    """Apply standard scaling to all features if required by the model"""
    # Only apply scaling if the model requires it
    if model_manager.requires_input_scaling():
        print("Applying standard scaling to all features")
        
        try:
            # Create DataFrame with columns in the exact order expected by scaler
            expected_features = standard_scaler.feature_names_in_
            scaler_df = pd.DataFrame(columns=expected_features)
            
            # Fill in values from data_dict where feature names match
            for feature in expected_features:
                if feature in data_dict:
                    scaler_df[feature] = [data_dict[feature]]
                else:
                    print(f"Warning: Missing feature for scaler: {feature}")
                    scaler_df[feature] = [0]  # Default value for missing features
            
            # Now apply scaling with columns in the correct order
            scaled_array = standard_scaler.transform(scaler_df)
            
            # Convert back to dict and update original data
            scaled_df = pd.DataFrame(scaled_array, columns=expected_features)
            scaled_dict = scaled_df.iloc[0].to_dict()
            data_dict.update(scaled_dict)
            
            print(f"Successfully scaled {len(expected_features)} features")
        except Exception as e:
            print(f"Warning: Scaling error: {str(e)}")
            print("Continuing with unscaled data")
    else:
        print("Scaling not applied -- not required by model")
        
    return data_dict

@app.get("/health")
async def health_check():
    """Health check endpoint for the API."""
    try:
        # Check if model is available
        model_manager.get_model()
        last_updated = datetime.fromtimestamp(model_manager.last_modified_time).isoformat()
        return {
            "status": "healthy", 
            "model": "active_model.pkl",
            "last_updated": last_updated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

# Endpoint for predictions
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model_type: str = "active",    
    current_user: dict = Depends(get_current_user)  # Add User Dependency
):
    try:
        user_id = current_user.get("user_id")
        
        # Get active model through the model manager
        try:
            model = model_manager.get_model()

            model_type = type(model).__name__

            # Get feature names based on model type
            if hasattr(model, 'feature_names_in_'):
                # scikit-learn models
                feature_names = model.feature_names_in_
            elif hasattr(model, 'feature_names_'):
                # CatBoost models
                feature_names = model.feature_names_
            else:
                # Fallback - try to get expected feature names from training data
                feature_names = X_train_reference.columns.tolist()
                print(f"Warning: Model does not expose feature names directly, using {len(feature_names)} features from training data")

            explainer = ShapExplainer(
                model=model,
                feature_names=feature_names, 
                training_data=X_train_reference
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Model not available: {str(e)}")

        # Convert request to dictionary
        input_data = request.dict()
        
        # Print original input data
        print("\n===== ORIGINAL INPUT DATA =====")
        print(json.dumps(input_data, indent=2, default=str))

        # Preprocess the input data using the saved preprocessors
        preprocessed_data = preprocess_input(input_data, model_type=model_type)

        # Print preprocessed data
        print("\n===== PREPROCESSED DATA =====")
        print(json.dumps(preprocessed_data, indent=2, default=str))

        scaled_preprocessed_data = scale_features(preprocessed_data)

        print("\n===== DATA AFTER SCALING =====")
        print(json.dumps(scaled_preprocessed_data, indent=2, default=str))

        # Create model input directly from scaled data
        # Get feature names based on model type
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        elif hasattr(model, 'feature_names_'):
            expected_features = model.feature_names_
        else:
            expected_features = X_train_reference.columns.tolist()
            print(f"Warning: Using feature names from training data for input creation")
        
        model_input = pd.DataFrame(columns=expected_features)

        # Fill model input from scaled data
        for col in expected_features:
            if col in scaled_preprocessed_data:
                model_input[col] = [scaled_preprocessed_data[col]]
            else:
                print(f"Warning: Missing feature for model: {col}")
                model_input[col] = [0]  # Default value for missing features

        # Log any missing or extra features for debugging
        missing_features = set(expected_features) - set(scaled_preprocessed_data.keys())
        extra_features = set(scaled_preprocessed_data.keys()) - set(expected_features)
        if missing_features:
            print(f"Warning: Features missing for model: {missing_features}")
        if extra_features:
            print(f"Warning: Extra features not used by model: {extra_features}")


        """
        # Print model's required features in a structured way
        print("\n===== MODEL REQUIRED FEATURES =====")
        print(f"Model type: {model_type}")
        print(f"Total features required: {len(expected_features)}")
        # Print all feature names, sorted alphabetically
        print("\nAll required feature names:")
        for i, feature in enumerate(sorted(expected_features)):
            print(f"  {i+1}. {feature}")

        """
        
        # Make prediction
        prediction = model.predict(model_input)[0]

        prediction_class = int(prediction)

        # Convert prediction to boolean and create approval message
        is_approved = bool(prediction == 1)
        approval_status = "Approved" if is_approved else "Rejected"

        # Calculate raw probability for class 1 (approval)
        raw_probability = float(model.predict_proba(model_input)[0][1])

        # Calculate confidence based on the predicted class
        if is_approved:
            # For approval (class 1), use probability directly
            confidence = raw_probability
        else:
            # For rejection (class 0), confidence is 1 - probability
            confidence = 1 - raw_probability

        # Format confidence once for consistent display
        confidence_formatted = str(round(confidence*100, 2)) + "%"

        # Print prediction results as debug message
        print("\n===== PREDICTION RESULTS =====")
        print(f"Predicted Loan Status: {approval_status} (Confidence: {confidence_formatted})")

        ##### XAI SHAP EXPLANATION #####
        # Generate explanation 
        explanation = None
        try:
            explanation = explainer.generate_explanation(model_input, prediction_class=prediction_class, auto_align_explanation=True)
        except Exception as e:
            print(f"Error generating explanation: {str(e)}") 
            # Use a default empty explanation structure or None
            explanation = {
                "summary_plot_featureimportance": "",
                "waterfall_plot": "",
                "force_plot": "",
                "top_features": [],
                "base_value": 0.0,
                "shap_values": [],
                "human_explanation":  {
                    "decision": approval_status,
                    "confidence": confidence_formatted,
                    "certainty_level": "unknown",
                    "positive_factors": [],
                    "negative_factors": []
                }
            }
                    
        # Get model info for response
        model_info = {
            "type": getattr(model, "_estimator_type", "unknown"),
            "last_updated": datetime.fromtimestamp(model_manager.last_modified_time).isoformat(),
            "features_count": len(expected_features)
        }

        # Return the prediction
        return PredictionResponse(
            loan_approval=is_approved,
            message=approval_status,
            confidence=confidence_formatted,
            model_info=model_info,
            explanation=explanation
        )   
    
    except Exception as e:
        # Log the error
        print(f"Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()  # This will give more detailed error information
        raise HTTPException(status_code=500, detail=f"Prediction ERROR: {str(e)}")
    
@app.post("/reload-models")
async def reload_models(current_user: dict = Depends(get_current_user)):
    """Force reload the active model"""
    try:
        # Check admin permissions
        if current_user.get("role") not in ["Admin", "AI Engineer"]:
            raise HTTPException(
                status_code=403, detail="Only admins and AI engineers can reload models"
            )
            
        # Force model manager to reload the model
        model_manager.last_modified_time = 0  # Reset last modified time
        success = model_manager.load_model()
        
        if success:
            model_info = {
                "type": getattr(model_manager.active_model, "_estimator_type", "unknown"),
                "last_updated": datetime.fromtimestamp(model_manager.last_modified_time).isoformat()
            }
            return {
                "success": True, 
                "message": "Model reloaded successfully", 
                "model_info": model_info
            }
        else:
            return {
                "success": False, 
                "error": "Failed to reload model"
            }
    except Exception as e:
        print(f"Error reloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")

@app.post("/update-reference-data")
async def update_reference_data(current_user: dict = Depends(get_current_user)):
    """
    Update the training data reference used for distance-based confidence calculation.
    This should be called after model retraining.
    """
    try:
        # Check admin permissions
        if current_user.get("role") not in ["Admin", "AI Engineer"]:
            raise HTTPException(
                status_code=403, detail="Only admins and AI engineers can update reference data"
            )
            
        # Path to the new reference data
        reference_data_path = "X_train.pkl"
        
        if os.path.exists(reference_data_path):
            try:
                global X_train_reference
                X_train_reference = joblib.load(reference_data_path)
                print(f"Training reference data updated successfully: {len(X_train_reference)} samples loaded")
                return {"success": True, "message": f"Reference data updated: {len(X_train_reference)} samples"}
            except Exception as e:
                print(f"Error updating reference data: {str(e)}")
                return {"success": False, "error": f"Error updating reference data: {str(e)}"}
        else:
            return {"success": False, "error": "Reference data file not found"}
    except Exception as e:
        print(f"Error in update-reference-data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8088, reload=True, log_level="debug")

# docker-compose build ml_api
# docker-compose up -d ml_api