from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

from .views import (
    # Static page views
    home, about, services, pricing, contact, documentation, api_documentation, fastapi_documentation,
    
    # Authentication views
    login_view, register_view, logout_view, password_reset, refresh_token_view, dashboard, user_profile, account_settings,
    
    # User management views
    user_management, add_user, edit_user, change_user_role, delete_user,
    
    # Prediction and feedback views
    prediction_form, submit_prediction_feedback, prediction_history, prediction_detail, prediction_feedback,
    
    # Model management views
    model_management, set_model_active, delete_model,
    
    # AI Engineer and Staff Review Predictions views
    review_predictions, staff_aiengineer_prediction_detail,
    
    # Admin analytics views
    admin_analytics, log_api_metrics, export_analytics_data,
)

# Group URLs by functional area for better organization
urlpatterns = [
    # Static Pages
    path('', home, name='home'),
    path('about/', about, name='about'),
    path('services/', services, name='services'),
    path('pricing/', pricing, name='pricing'),
    path('documentation/', documentation, name='documentation'),
    path('api/', api_documentation, name='api_documentation'),
    path('fastapi/', fastapi_documentation, name='fastapi_documentation'),
    path('contact/', contact, name='contact'),
    
    # Authentication URLs
    path('login/', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('logout/', logout_view, name='logout'),
    path('reset-password/', password_reset, name='password_reset'),
    path('refresh-token/', refresh_token_view, name='refresh_token'),
    path('dashboard/', dashboard, name='dashboard'),
    path('profile/', user_profile, name='user_profile'),
    path('settings/', account_settings, name='account_settings'),
    
    # User Management URLs
    path('user-management/', user_management, name='user_management'),
    path('user-management/add/', add_user, name='add_user'),
    path('user-management/edit/<int:user_id>/', edit_user, name='edit_user'),
    path('user-management/change-role/<int:user_id>/', change_user_role, name='change_user_role'),
    path('user-management/delete/<int:user_id>/', delete_user, name='delete_user'),
    
    # Prediction and Feedback URLs
    path('predict/', prediction_form, name='prediction_form'),
    path('predict/feedback/', submit_prediction_feedback, name='submit_prediction_feedback'),
    path('predict/history/', prediction_history, name='prediction_history'),
    path('predict/detail/<int:prediction_id>/', prediction_detail, name='prediction_detail'),
    path('predict/feedback/<int:prediction_id>/', prediction_feedback, name='prediction_feedback'),
    
    # Model Management URLs
    path('model-management/', model_management, name='model_management'),
    path('set-model-active/<int:model_id>/', set_model_active, name='set_model_active'),
    path('delete-model/<int:model_id>/', delete_model, name='delete_model'),
    
    # AI Engineer and Staff to review predictions URLs
    path('review-predictions/', review_predictions, name='review_predictions'),
    path('prediction-detail/<int:prediction_id>/', staff_aiengineer_prediction_detail, name='staff_aiengineer_prediction_detail'),    
         
    # Admin Analytics URLs - Changed from 'admin/analytics/' to 'analytics/' to avoid conflict
    path('analytics/', admin_analytics, name='admin_analytics'),
    path('analytics/export/', export_analytics_data, name='export_analytics_data'),
    path('api/log-metrics/', log_api_metrics, name='log_api_metrics'),
]
