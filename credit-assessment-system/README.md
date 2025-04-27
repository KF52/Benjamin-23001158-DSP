# XAI Loan Credit Assessment System
BENJAMIN KHOR 
23001158 
UFCFXK-30-3 - Digital systems project

## Project Overview
This project is an Explainable AI (XAI) powered Loan Credit Assessment System designed to provide transparent, interpretable, and automated loan approval decisions. The system uses machine learning models to analyze applicant data and predict loan approval outcomes while providing clear explanations for these decisions using SHAP (SHapley Additive exPlanations) technology. This project is intended for production use in the financial sector.

Core functionalities include:

- **Loan Approval Prediction**: Analyzes applicant financial and personal data to determine loan eligibility using machine learning models
- **Transparent Decision Making**: Provides detailed explanations of factors influencing loan decisions using SHAP technology
- **Multi-model Support**: Integrates various ML models (CatBoost, XGBoost, etc.) with ability to compare performance and switch between them
- **Applicant Dispute System**: Allows applicants to challenge decisions for reconsideration
- **Interactive Visualization**: Visual representations of feature importance and model decisions
- **Role-based Access**:
  - **End Users**: Submit loan applications and view personalized explanations
  - **Staff**: Review applications, access records, and override automated decisions when necessary
  - **AI Engineers**: Upload new models, set other exisitng models as active
  - **Administrators**: Manage user accounts, adjust system settings, and monitor system performance
- **Model Management**: Complete toolkit for managing ML models including uploading, versioning, monitoring, and activation
- **User Administration**: Privilege to create, update, or delete user accounts and manage permission levels

The system bridges the gap between complex AI decision-making and human understanding, helping both lenders make informed lending decisions and borrowers understand their creditworthiness while maintaining compliance with regulatory requirements for algorithmic transparency in financial services.

## Tech Stack

**Frontend/UI:**
- Bootstrap 5 framework
- Interactive JavaScript components
- Chart.js for data visualization

**Backend:**
- Django
- FastAPI for ML microservice
- PostgreSQL database
- JWT authentication

**ML Models:**
- CatBoost Classifier

**Explainable AI:**
- SHAP (SHapley Additive exPlanations)

## Installation and Setup

```bash
# Clone the repository
git clone # <project repository link>
cd loan-credit-assessment-system

# Build and start all services using Docker Compose
docker-compose build
docker-compose up -d

# Access the services using the link:
# - Django web app: http://localhost:8000
```

## Usage
1. Register for an account on the platform 
*First user will be automatically be assigned with an admin role, and the next folllowing user will be assigned with user role.
2. Navigate to the role-dedicated dashboard after login.
3. For loan prediction:
   - Fill out the loan application form with required financial details
   - Submit the form for AI analysis
   - Review the prediction result and explanation
   - View detailed breakdown of factors influencing the decision
 