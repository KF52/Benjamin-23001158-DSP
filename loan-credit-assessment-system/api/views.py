from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.utils import timezone
from django.urls import reverse
from django.db.models import Count, Sum, Q
from django.core.paginator import Paginator
from django.conf import settings

from .models import User, Prediction, MLModel, APIMetrics
from .serializers import PredictionSerializer
from utils.ml_api_client import MLApiClient
import uuid
import json
import logging

from django.http import HttpResponse,response,JsonResponse,HttpResponseBadRequest
from .models import Role, Prediction, User, MLModel, APIMetrics

import re
import stripe
from datetime import timedelta
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .models import User, Role, Prediction, PredictionUpdateForm
from .permissions import IsAdminUser
from utils.ml_api_client import predict, health
from django.utils import timezone
import re

import json
import os
import joblib
import shutil
from django.contrib import messages
from datetime import datetime, timedelta
from django.core.paginator import Paginator
import json
from datetime import timedelta, datetime
from django.db.models import Count, Sum, Avg, F, ExpressionWrapper, fields
from django.db.models.functions import TruncDay, TruncWeek, TruncMonth

#########################################################################
# GENERAL PAGE VIEWS
#########################################################################

def home(request):
    """Rendering the home page"""
    # If user is already logged in, redirect to dashboard
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request,'general/home.html')

def about(request):
    """About page view"""
    return render(request,'general/about.html')

def services(request):
    """Services page view"""
    return render(request,'general/services.html')

def pricing(request):
    """Pricing page view"""
    return render(request,'general/pricing.html')

def documentation(request):
    """Documentation page view"""
    return render(request,'general/documentation.html')

def api_documentation(request):
    """API Documentation page view"""
    return render(request,'general/api_documentation.html')

def fastapi_documentation(request):
    """FastAPI Documentation page view"""
    return render(request,'general/fastapi_documentation.html')

def contact(request):
    """Contact page view"""
    return render(request,'general/contact.html')

#########################################################################
# AUTHENTICATION VIEWS
#########################################################################

def register_view(request):
    """Registration view function"""
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        # Basic validation
        if not all([name, email, password, confirm_password]):
            return render(request, 'auth/register.html', {'error_message': 'All fields are required'})
        
        if password != confirm_password:
            return render(request, 'auth/register.html', {'error_message': 'Passwords do not match'})
        
        # Check password strength
        if len(password) < 8:
            return render(request, 'auth/register.html', {'error_message': 'Password must be at least 8 characters long'})
        
        # Check if email already exists
        if User.objects.filter(email=email).exists():
            return render(request, 'auth/register.html', {'error_message': 'Email is already registered'})
        
        # Create new user
        try:
            user = User.objects.create_user(email=email, name=name, password=password)
            user.save()
            
            # Automatically log in the user
            login(request, user)
            
            # Redirect to home page or display success message
            return redirect('home')
        except Exception as e:
            return render(request, 'auth/register.html', {'error_message': f'Registration error: {str(e)}'})
    
    return render(request, 'auth/register.html')

def login_view(request):
    """Login view function"""
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        # Basic validation
        if not all([email, password]):
            return render(request, 'auth/login.html', {'error_message': 'Email and password are required'})
        
        # Authenticate user
        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            # Log in the user
            login(request, user)
            
            # Generate JWT token
            refresh = RefreshToken.for_user(user)
            access_token = str(refresh.access_token)
            
            # Prepare the response
            response = redirect('dashboard')
            
            # Set HttpOnly cookie with the JWT token (for development)
            response.set_cookie(
                key='access_token',
                value=access_token,
                httponly=True,   
                secure=False,    # Since Running LocalHost No Need for SSL/TLS
                samesite='Lax',  
                max_age=3600     # 1 hour expiry (Seconds)
            )
            
            # Set refresh token as well
            response.set_cookie(
                key='refresh_token',
                value=str(refresh),
                httponly=True,
                secure=False,    
                samesite='Lax',
                max_age=86400    # 1 day expiry
            )
            
            return response
        else:
            return render(request, 'auth/login.html', {'error_message': 'Invalid email or password'})
    
    return render(request, 'auth/login.html')

def logout_view(request):
    """Logout view function"""
    # Django Logout 
    logout(request)
    
    # Create response object
    response = redirect('home')
    
    # Deleting the cookies
    response.delete_cookie('access_token')
    response.delete_cookie('refresh_token')
    
    return response

def password_reset(request):
    """Password Reset Mechanism using Email and FullName"""
    if request.method == 'POST':
        email = request.POST.get('email')
        full_name = request.POST.get('full_name')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')
        
        # Basic validation
        if not all([email, full_name, new_password, confirm_password]):
            return render(request, 'auth/password_reset.html', {
                'error_message': 'All fields are required'
            })
        
        if new_password != confirm_password:
            return render(request, 'auth/password_reset.html', {
                'error_message': 'Passwords do not match'
            })
        
        if len(new_password) < 8:
            return render(request, 'auth/password_reset.html', {
                'error_message': 'Password must be at least 8 characters long'
            })
        
        # Try to find the user
        try:
            user = User.objects.get(email=email)
            
            # Verify full name
            if user.name != full_name:
                return render(request, 'auth/password_reset.html', {
                    'error_message': 'The information you provided does not match our records'
                })
            
            # Reset the password
            user.set_password(new_password)
            user.save()
            
            return render(request, 'auth/password_reset.html', {
                'success_message': 'Your password has been reset successfully. You can now log in with your new password.'
            })
            
        except User.DoesNotExist:
            # For security, don't reveal that the user doesn't exist
            return render(request, 'auth/password_reset.html', {
                'error_message': 'The information you provided does not match our records'
            })
    
    return render(request, 'auth/password_reset.html')

def refresh_token_view(request):
    """Handle refresh token to get a new access token"""
    refresh_token = request.COOKIES.get('refresh_token')
    if not refresh_token:
        return JsonResponse({'error': 'Refresh token not found'}, status=401)
    
    try:
        refresh = RefreshToken(refresh_token)
        access_token = str(refresh.access_token)
        
        response = JsonResponse({'success': True})
        response.set_cookie(
            key='access_token',
            value=access_token,
            httponly=True,
            secure=settings.COOKIE_SECURE,  # Environment-aware
            samesite='Lax',
            max_age=3600
        )
        
        return response
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=401)

#########################################################################
# DASHBOARD VIEWS
#########################################################################

@login_required
def dashboard(request):
    """Main dashboard view"""
    user = request.user
    # Base context with user info
    context = {
        'user': user,
        'role': user.role,
    }
    
    # Add recent activities based on user role
    recent_activities = []
    
    if user.role == 'Admin':
        # Add admin-specific data
        context.update({
            'total_users': User.objects.count(),
            'recent_registrations': User.objects.order_by('-member_since')[:5]
        })
        
        # Recent users (new registrations)
        recent_users = User.objects.order_by('-member_since')[:5]
        for user_item in recent_users:
            recent_activities.append({
                'type': 'user_registration',
                'content': f"New user registered: {user_item.name}",
                'timestamp': user_item.member_since,
            })

        # Recent API metrics (errors)
        recent_api_errors = APIMetrics.objects.filter(error=True).order_by('-timestamp')[:5]
        for error in recent_api_errors:
            recent_activities.append({
                'type': 'api_error',
                'content': f"API Error: {error.endpoint} (Status {error.status_code})",
                'timestamp': error.timestamp,
            })

    elif user.role == 'Staff':
        # Add staff-specific data
        context = {
            'pending_reviews': Prediction.objects.filter(is_checked=False).count(),
            'recent_predictions': Prediction.objects.order_by('-timestamp')[:5],
            'user_count': User.objects.filter(role=Role.END_USER).count()
        }
        
        # Recent prediction activities for staff dashboard
        recent_predictions = Prediction.objects.order_by('-timestamp')[:10]
        for prediction in recent_predictions:
            approval_status = 'Approved' if prediction.loan_approval else 'Rejected'
            recent_activities.append({
                'type': 'prediction',
                'content': f"Loan#{prediction.id} by {prediction.user.name} - Status: {approval_status}",
                'timestamp': prediction.timestamp,
            })
        
        # Recent loan status overrides by staff
        recent_overrides = Prediction.objects.filter(
            modified_by_staff=True
        ).order_by('-modified_at')[:5]  # Get 5 most recent overrides

        for prediction in recent_overrides:
            original_status = 'Approved' if prediction.original_prediction else 'Rejected'
            new_status = 'Approved' if prediction.loan_approval else 'Rejected'
            staff_name = prediction.modified_by.name if prediction.modified_by else 'Unknown Staff'
            
            recent_activities.append({
                'type': 'override',  # New activity type for overrides
                'content': f"Loan#{prediction.id} status changed from [{original_status}] to [{new_status}] by {staff_name}",
                'timestamp': prediction.modified_at,  # Use the override timestamp
            })

    elif user.role == 'AI Engineer':
        # Add AI engineer-specific data
        recent_models = MLModel.objects.order_by('-upload_date')[:5]
        
        # Get count of unchecked predictions
        unchecked_count = Prediction.objects.filter(is_checked=False).count()
        
        context.update({
            'model_count': MLModel.objects.count(),
            'unchecked_predictions': unchecked_count
        })
        
        # Recent model uploads
        for model in recent_models:
            status = "active" if model.is_active else "inactive"
            recent_activities.append({
                'type': 'model',
                'content': f"MODEL {model.model_name} ({model.model_type}) uploaded - {status}",
                'timestamp': model.upload_date,
            })
            
        # Recent predictions that need review
        recent_predictions = Prediction.objects.filter(is_checked=False).order_by('-timestamp')[:5]
        for pred in recent_predictions:
            approval_status = 'Approved' if pred.loan_approval else 'Rejected'
            recent_activities.append({
                'type': 'prediction',
                'content': f"Loan#{pred.id} by {pred.user.name} - Status: ({approval_status}) NEEDS REVIEW",
                'timestamp': pred.timestamp,
            })
            
    else:
        # Add regular user-specific data
        user_predictions = Prediction.objects.filter(user=user).order_by('-timestamp')[:5]
        
        # User's recent predictions
        for pred in user_predictions:
            approval_status = 'Approved' if pred.loan_approval else 'Rejected'
            recent_activities.append({
                'type': 'prediction',
                'content': f"Loan#{pred.id} by {pred.user.name} - Status: {approval_status}",
                'timestamp': pred.timestamp,
            })

    # Sort all activities by timestamp (newest first)
    recent_activities.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Add timestamp display for all activities
    for activity in recent_activities:
        # Calculate time difference
        time_diff = timezone.now() - activity['timestamp']
        days = time_diff.days
        
        if days == 0:
            hours = time_diff.seconds // 3600
            if hours == 0:
                minutes = time_diff.seconds // 60
                activity['time_ago'] = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                activity['time_ago'] = f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif days == 1:
            activity['time_ago'] = "Yesterday"
        else:
            activity['time_ago'] = f"{days} days ago"
    
    # Add debug for checking activity count
    print(f"DEBUG: Generated {len(recent_activities)} activities for user {user.name} with role {user.role}")
    if recent_activities:
        print(f"DEBUG: First activity: {recent_activities[0]['content']} from {recent_activities[0]['timestamp']}")
    
    # Limit to top 10 activities
    context['recent_activities'] = recent_activities[:10]
    
    # Debug context to ensure we're passing activities to template
    print(f"DEBUG: Context has recent_activities: {'recent_activities' in context}")
    print(f"DEBUG: Number of activities in context: {len(context.get('recent_activities', []))}")
    
    return render(request, 'dashboard/dashboard.html', context)

#########################################################################
# USER MANAGEMENT VIEWS (ADMIN)
#########################################################################

@login_required
def user_management(request):
    """View for user management page (admin only)"""
    # Check if user is admin
    if request.user.role != Role.ADMIN:
        messages.error(request, "Access denied. Admin privileges required.")
        return redirect('dashboard')
    
    # Get all users
    users = User.objects.all().order_by('name')
    
    return render(request, 'dashboard/user_management.html', {'users': users})

@login_required
def add_user(request):
    """Add a new user (admin only)"""
    if request.user.role != Role.ADMIN:
        messages.error(request, "Access denied. Admin privileges required.")
        return redirect('dashboard')
    
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        role = request.POST.get('role')
        
        # Validate input
        if not all([name, email, password, role]):
            messages.error(request, "All fields are required.")
            return redirect('user_management')
        
        if len(password) < 8:
            messages.error(request, "Password must be at least 8 characters long.")
            return redirect('user_management')
        
        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "A user with this email already exists.")
            return redirect('user_management')
        
        # Create user
        try:
            user = User.objects.create_user(email=email, name=name, password=password, role=role)
            user.save()
            messages.success(request, f"User {name} created successfully.")
        except Exception as e:
            messages.error(request, f"Error creating user: {str(e)}")
        
        return redirect('user_management')
    
    return redirect('user_management')

@login_required
def edit_user(request, user_id):
    """Edit a user (admin only)"""
    if request.user.role != Role.ADMIN:
        messages.error(request, "Access denied. Admin privileges required.")
        return redirect('dashboard')
    
    user_to_edit = get_object_or_404(User, id=user_id)
    
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        
        # Validate input
        if not all([name, email]):
            messages.error(request, "Name and email are required.")
            return redirect('user_management')
        
        # Check if email exists and belongs to another user
        if User.objects.filter(email=email).exclude(id=user_id).exists():
            messages.error(request, "A user with this email already exists.")
            return redirect('user_management')
        
        # Update user
        try:
            user_to_edit.name = name
            user_to_edit.email = email
            user_to_edit.save()
            messages.success(request, f"User {name} updated successfully.")
        except Exception as e:
            messages.error(request, f"Error updating user: {str(e)}")
        
        return redirect('user_management')
    
    return redirect('user_management')

@login_required
def change_user_role(request, user_id):
    """Change a user's role (admin only)"""
    if request.user.role != Role.ADMIN:
        messages.error(request, "Access denied. Admin privileges required.")
        return redirect('dashboard')
    
    user_to_change = get_object_or_404(User, id=user_id)
    
    if request.method == 'POST':
        role = request.POST.get('role')
        
        # Validate input
        if not role:
            messages.error(request, "Role is required.")
            return redirect('user_management')
        
        # Don't allow changing the last admin
        if user_to_change.role == Role.ADMIN and role != Role.ADMIN:
            admin_count = User.objects.filter(role=Role.ADMIN).count()
            if admin_count <= 1:
                messages.error(request, "Cannot change the role of the last admin.")
                return redirect('user_management')
        
        # Update user role
        try:
            user_to_change.role = role
            user_to_change.save()
            messages.success(request, f"Role for {user_to_change.name} changed to {role} successfully.")
        except Exception as e:
            messages.error(request, f"Error changing role: {str(e)}")
        
        return redirect('user_management')
    
    return redirect('user_management')

@login_required
def delete_user(request, user_id):
    """Delete a user (admin only)"""
    if request.user.role != Role.ADMIN:
        messages.error(request, "Access denied. Admin privileges required.")
        return redirect('dashboard')
    
    user_to_delete = get_object_or_404(User, id=user_id)
    
    # Don't allow deleting yourself
    if user_to_delete.id == request.user.id:
        messages.error(request, "You cannot delete your own account.")
        return redirect('user_management')
    
    # Don't allow deleting the last admin
    if user_to_delete.role == Role.ADMIN:
        admin_count = User.objects.filter(role=Role.ADMIN).count()
        if admin_count <= 1:
            messages.error(request, "Cannot delete the last admin.")
            return redirect('user_management')
    
    if request.method == 'POST':
        try:
            name = user_to_delete.name
            user_to_delete.delete()
            messages.success(request, f"User {name} deleted successfully.")
        except Exception as e:
            messages.error(request, f"Error deleting user: {str(e)}")
        
        return redirect('user_management')
    
    return redirect('user_management')

#########################################################################
# USER PROFILE VIEW
#########################################################################

@login_required
def user_profile(request):
    """View for user profile page"""
    user = request.user
    
    # Get user's prediction statistics
    prediction_count = Prediction.objects.filter(user=user).count()
    recent_predictions = Prediction.objects.filter(user=user).order_by('-timestamp')[:5]
    
    context = {
        'user': user,
        'prediction_count': prediction_count,
        'recent_predictions': recent_predictions,
    }
    
    return render(request, 'auth/profile.html', context)

@login_required
def account_settings(request):
    """View for account settings page"""
    user = request.user
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'update_profile':
            # Update basic profile information
            name = request.POST.get('name')
            email = request.POST.get('email')
            
            # Validate input
            if not all([name, email]):
                messages.error(request, "Name and email are required.")
                return redirect('account_settings')
            
            # Check if email exists and belongs to another user
            if User.objects.filter(email=email).exclude(id=user.id).exists():
                messages.error(request, "This email is already in use by another account.")
                return redirect('account_settings')
            
            # Update user
            try:
                user.name = name
                user.email = email
                user.save()
                messages.success(request, "Profile updated successfully.")
            except Exception as e:
                messages.error(request, f"Error updating profile: {str(e)}")
            
        elif action == 'change_password':
            # Change password
            current_password = request.POST.get('current_password')
            new_password = request.POST.get('new_password')
            confirm_password = request.POST.get('confirm_password')
            
            # Validate input
            if not all([current_password, new_password, confirm_password]):
                messages.error(request, "All password fields are required.")
                return redirect('account_settings')
            
            if new_password != confirm_password:
                messages.error(request, "New passwords don't match.")
                return redirect('account_settings')
            
            if len(new_password) < 8:
                messages.error(request, "Password must be at least 8 characters long.")
                return redirect('account_settings')
            
            # Check current password
            if not user.check_password(current_password):
                messages.error(request, "Current password is incorrect.")
                return redirect('account_settings')
            
            # Update password
            try:
                user.set_password(new_password)
                user.save()
                # Re-authenticate user to prevent logout
                updated_user = authenticate(request, email=user.email, password=new_password)
                if updated_user:
                    login(request, updated_user)
                messages.success(request, "Password changed successfully.")
            except Exception as e:
                messages.error(request, f"Error changing password: {str(e)}")
        
        elif action == 'notification_preferences':
            # Update notification preferences (can be extended in the future)
            email_notifications = request.POST.get('email_notifications') == 'on'
            
            # For now, just show a success message
            messages.success(request, "Notification preferences updated successfully.")
        
        # Add other account settings actions as needed
        
        return redirect('account_settings')
    
    return render(request, 'auth/account_settings.html', {'user': user})

#########################################################################
# PREDICTION VIEWS - USER
#########################################################################

@login_required
def prediction_form(request):
    """View for the prediction form"""
    # Check if ML service is available
    ml_service_available = health()
    
    if request.method == 'POST':
        # Extract form data and convert to expected format
        input_data = {}
        
        # Process each field in the form - convert strings to appropriate types
        for key, value in request.POST.items():
            if key in ['csrfmiddlewaretoken']:
                continue
                
            # Convert boolean fields
            if value.lower() == 'true':
                input_data[key] = True
            elif value.lower() == 'false':
                input_data[key] = False
            # Convert numeric fields
            elif value and value.replace('.', '', 1).isdigit():
                input_data[key] = float(value)
            else:
                input_data[key] = value
        
        try:
            # Make prediction using client
            prediction_result = predict(input_data, request=request)
            
            loan_approval = prediction_result.get('loan_approval', False)
            confidence_score = prediction_result.get('probability', 0.0)

            # Ensure explanation is included in the result
            explanation = prediction_result.get('explanation')
            
            # If explanation is separate, make sure to include it in result
            if explanation and 'explanation' not in prediction_result:
                prediction_result['explanation'] = explanation
            
            # Automatically save the prediction
            prediction = Prediction(
                user=request.user,
                input_data=input_data,
                result=prediction_result,
                loan_approval=loan_approval,
                confidence_score=confidence_score,
            )
            prediction.save()
            
            # Render the result page with the prediction ID
            return render(request, 'predictions/prediction_result.html', {
                'prediction': prediction_result,
                'input_data': input_data,
                'prediction_id': prediction.id,
                'explanation': explanation
            })
            
        except Exception as e:
            # Log the error
            print(f"Prediction error: {str(e)}")
            
            # Render the form again with an error message
            return render(request, 'predictions/prediction_form.html', {
                'error_message': f"Error making prediction: {str(e)}",
                'ml_service_available': False
            })
    
    # Render the initial form
    return render(request, 'predictions/prediction_form.html', {
        'ml_service_available': ml_service_available
    })

@login_required
def prediction_history(request):
    """View for user's prediction history"""
    # Get the user's predictions
    predictions = Prediction.objects.filter(user=request.user)
    
    return render(request, 'predictions/prediction_history.html', {
        'predictions': predictions
    })

@login_required
def prediction_detail(request, prediction_id):
    """View for detailed prediction information"""
    try:
        prediction = Prediction.objects.get(id=prediction_id, user=request.user)

        # Extract explanation from result
        explanation_dict = prediction.result.get('explanation', {}) if prediction.result else {}
        
        # Create an explanation context with all visualization components
        explanation = {
            'waterfall_plot': explanation_dict.get('waterfall_plot'),
            'summary_plot_featureimportance': explanation_dict.get('summary_plot_featureimportance'),
            'force_plot': explanation_dict.get('force_plot'),
            'top_features': explanation_dict.get('top_features', []),
            'base_value': explanation_dict.get('base_value'),
            'feature_impact': explanation_dict.get('feature_impact', []),
            'log_odds': explanation_dict.get('log_odds', 0.0),
            'probability': explanation_dict.get('probability', 0.0),
            'probability_percentage': explanation_dict.get('probability_percentage', 0.0),
            'shap_values': explanation_dict.get('shap_values', []),
            'prediction_class': 1 if prediction.loan_approval else 0,
            'human_explanation': explanation_dict.get('human_explanation', {
                'decision': 'Unknown',
                'confidence': '0%',
                'certainty_level': 'unknown',
                'positive_factors': [],
                'negative_factors': []
            })
        }
        
        return render(request, 'predictions/prediction_detail.html', {
            'prediction': prediction,
            'input_data': prediction.input_data,
            'result': prediction.result,
            'explanation': explanation
        })
    except Prediction.DoesNotExist:
        messages.error(request, "Prediction not found or you don't have permission to access it.")
        return redirect('prediction_history')

@login_required
def prediction_feedback(request, prediction_id):
    """View for providing feedback on a prediction"""
    try:
        prediction = Prediction.objects.get(id=prediction_id, user=request.user)
        
        # Check if feedback has already been provided
        if prediction.is_satisfactory is not None:
            messages.warning(request, "Feedback has already been provided for this prediction.")
            return redirect('prediction_detail', prediction_id=prediction.id)
        
        if request.method == 'POST':
            is_satisfactory = request.POST.get('is_satisfactory') == 'yes'
            
            prediction.is_satisfactory = is_satisfactory
            prediction.feedback_date = timezone.now()

            if not is_satisfactory:
                applicant_comments = request.POST.get('applicant_comments')
                
                if not applicant_comments:
                    messages.error(request, "Please provide your feedback and our bank officer will contact you soon.")
                    return render(request, 'predictions/prediction_feedback.html', {'prediction': prediction})
                
                prediction.applicant_comments = applicant_comments
                prediction.needs_review = True
                
                messages.info(request, "Your feedback has been recorded. This case has been flagged for review.")
            else:
                messages.success(request, "Thank you for confirming this loan application.")
            
            prediction.save()
            return redirect('prediction_history')
        
        return render(request, 'predictions/prediction_feedback.html', {
            'prediction': prediction,
            'input_data': prediction.input_data,
            'result': prediction.result
        })
        
    except Prediction.DoesNotExist:
        messages.error(request, "Prediction not found or you don't have permission to access it.")
        return redirect('prediction_history')

@login_required
def submit_prediction_feedback(request):
    """Handle submission of feedback on a prediction"""
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method")
    
    prediction_id = request.POST.get('prediction_id')
    is_satisfactory = request.POST.get('is_satisfactory') == 'yes'
    
    try:
        prediction = Prediction.objects.get(id=prediction_id, user=request.user)
        
        # Update the prediction with feedback
        prediction.is_satisfactory = is_satisfactory
        prediction.feedback_date = timezone.now()
        
        if not is_satisfactory:
            applicant_comments = request.POST.get('applicant_comments')
            
            if not applicant_comments:
                messages.error(request, "Please provide your feedback and our bank officer will contact you soon.")
                return redirect('prediction_result', prediction_id=prediction_id)
            
            prediction.applicant_comments = applicant_comments
            prediction.needs_review = True
            
            messages.info(request, "Your feedback has been recorded. This case has been flagged for review.")
        else:
            messages.success(request, "Thank you for confirming this loan application.")
        
        prediction.save()
        
        # Redirect to prediction history
        return redirect('prediction_history')
        
    except Prediction.DoesNotExist:
        messages.error(request, "Prediction not found or you don't have permission to access it.")
        return redirect('dashboard')
    except ValueError:
        return redirect('prediction_result', prediction_id=prediction_id)
    except Exception as e:
        messages.error(request, f"Error processing feedback: {str(e)}")
        return redirect('prediction_history')

#########################################################################
# PREDICTION VIEWS - AI ENGINEER and STAFF
#########################################################################

@login_required
def review_predictions(request):
    """View for AI Engineers to review all user predictions"""
    # Check if user is AI Engineer or Admin
    if request.user.role not in ['AI Engineer', 'Admin', 'Staff']:
        messages.error(request, "Access denied")
        return redirect('dashboard')
    
    # Store the user role for template to adapt display
    user_role = request.user.role
    
    # Handle marking prediction as checked
    if request.method == 'POST' and 'prediction_id' in request.POST:
        prediction_id = request.POST.get('prediction_id')
        try:
            prediction = Prediction.objects.get(id=prediction_id)
            
            # Toggle the checked status
            prediction.is_checked = not prediction.is_checked
            
            # If we're checking it, also clear the needs_review flag
            if prediction.is_checked and prediction.needs_review:
                prediction.needs_review = False
                
            prediction.save()
            
            if prediction.is_checked:
                messages.success(request, f"Prediction #{prediction_id} marked as checked.")
            else:
                messages.info(request, f"Prediction #{prediction_id} unmarked.")
            
            return redirect('review_predictions')
        except Prediction.DoesNotExist:
            messages.error(request, "Prediction not found.")
    
    # Get filter parameters
    status_filter = request.GET.get('status', '')
    
    # Get all predictions, ordered by newest first
    predictions = Prediction.objects.all().select_related('user').order_by('-timestamp')
    
    # Apply filters
    if status_filter == 'checked':
        predictions = predictions.filter(is_checked=True)
    elif status_filter == 'unchecked':
        predictions = predictions.filter(is_checked=False)
    elif status_filter == 'disputed':
        predictions = predictions.filter(is_satisfactory=False)
    elif status_filter == 'needs_review':
        predictions = predictions.filter(needs_review=True)
    elif status_filter == 'approved':
        predictions = predictions.filter(loan_approval=True)
    elif status_filter == 'rejected':
        predictions = predictions.filter(loan_approval=False)
    elif status_filter == 'overridden':
        predictions = predictions.filter(modified_by_staff=True)
    
    # Pagination
    paginator = Paginator(predictions, 20)  # Show 20 predictions per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'predictions/review_predictions.html', {
        'page_obj': page_obj,
        'total_predictions': predictions.count(),
        'checked_count': predictions.filter(is_checked=True).count(),
        'unchecked_count': predictions.filter(is_checked=False).count(),
        'disputed_count': predictions.filter(is_satisfactory=False).count(),
        'needs_review_count': predictions.filter(needs_review=True).count(),
        'status_filter': status_filter,
        'user_role': user_role  # Pass user role to template

    })

@login_required
def staff_aiengineer_prediction_detail(request, prediction_id):
    """View for AI Engineers to see detailed prediction information"""
    if request.user.role not in ['Staff', 'AI Engineer', 'Admin']:
        messages.error(request, "Access denied.")
        return redirect('dashboard')
    
    try:
        prediction = Prediction.objects.get(id=prediction_id)

        # Store original prediction value when first viewing
        if prediction.original_prediction is None:
            prediction.original_prediction = prediction.loan_approval
            prediction.save()

        # Extract explanation from result
        explanation_dict = prediction.result.get('explanation', {}) if prediction.result else {}
        
        # Create an explanation context with all visualization components
        explanation = {
            'waterfall_plot': explanation_dict.get('waterfall_plot'),
            'summary_plot_featureimportance': explanation_dict.get('summary_plot_featureimportance'),
            'force_plot': explanation_dict.get('force_plot'),
            'top_features': explanation_dict.get('top_features', []),
            'base_value': explanation_dict.get('base_value'),
            'feature_impact': explanation_dict.get('feature_impact', []),
            'log_odds': explanation_dict.get('log_odds', 0.0),
            'probability': explanation_dict.get('probability', 0.0),
            'probability_percentage': explanation_dict.get('probability_percentage', 0.0),
            'shap_values': explanation_dict.get('shap_values', []),
            'prediction_class': 1 if prediction.loan_approval else 0,
            'human_explanation': explanation_dict.get('human_explanation', {
                'decision': 'Unknown',
                'confidence': '0%',
                'certainty_level': 'unknown',
                'positive_factors': [],
                'negative_factors': []
            })
        }
        
        # Handle marking prediction as checked
        if request.method == 'POST':
            if 'mark_checked' in request.POST:
                prediction.is_checked = True
                if prediction.needs_review:
                    prediction.needs_review = False
                prediction.save()
                messages.success(request, f"Prediction #{prediction_id} marked as checked.")
                return redirect('aiengineer_prediction_detail', prediction_id=prediction_id)
            
            elif 'update_status' in request.POST:
                # Get the original form data
                form = PredictionUpdateForm(request.POST, instance=prediction)
                if form.is_valid():
                    # Check if the loan_approval is being changed
                    if 'loan_approval' in form.changed_data:
                        update = form.save(commit=False)
                        update.modified_by_staff = True
                        update.modified_by = request.user
                        update.modified_at = timezone.now()
                        update.save()
                        messages.success(request, f"Loan status for prediction #{prediction_id} updated successfully.")
                    else:
                        # Just save comments without marking as staff-modified
                        form.save()
                        messages.success(request, f"Comments updated for prediction #{prediction_id}.")
                    return redirect('staff_aiengineer_prediction_detail', prediction_id=prediction_id)
        
        # create form for status update
        update_form = PredictionUpdateForm(instance=prediction)

        return render(request, 'predictions/staff_aiengineer_prediction_detail.html', {
            'prediction': prediction,
            'input_data': prediction.input_data,
            'result': prediction.result,
            'explanation': explanation,
            'user': prediction.user,  # Pass user info to template
            'user_role': request.user.role,  # Pass user role to template
            'approval_status': "Approved" if prediction.loan_approval else "Rejected",
            'confidence_score': float(prediction.confidence_score) * 100 if hasattr(prediction, 'confidence_score') else 0,
        })
    except Prediction.DoesNotExist:
        messages.error(request, "Prediction not found.")
        return redirect('review_predictions')


#########################################################################
# ML MODEL MANAGEMENT VIEWS
#########################################################################

@login_required
def model_management(request):
    """View for AI Engineers to manage ML models"""
    # Check if user is AI Engineer
    if request.user.role != 'AI Engineer' and request.user.role != 'Admin':
        messages.error(request, "Access denied. AI Engineer privileges required.")
        return redirect('dashboard')
    
    # Handle form submission for model upload
    if request.method == 'POST':
        model_file = request.FILES.get('model_file')
        model_name = request.POST.get('model_name')
        model_type = request.POST.get('model_type')
        description = request.POST.get('description', '')
        set_active = request.POST.get('set_active') == 'on'
        
        # Validate required fields
        if not all([model_file, model_name, model_type]):
            messages.error(request, "Please provide all required fields: model file, name, and type.")
            return redirect('model_management')
        
        # Validate file is .pkl or .h5
        if not model_file.name.endswith(('.pkl', '.h5')):
            messages.error(request, "Only .pkl and .h5 files are allowed.")
            return redirect('model_management')

        requires_scaling = 'requires_scaling' in request.POST
        
        # Save model with provided info
        model = MLModel(
            model_name=model_name,
            model_type=model_type,
            description=description,
            file=model_file,
            uploaded_by=request.user,
            is_active=set_active,
            requires_scaling=requires_scaling
        )
        model.save()
        
        # If set as active, copy to FastAPI directory
        if set_active:
            try:
                # Generate destination filename with the same extension as the uploaded file
                file_extension = os.path.splitext(model_file.name)[1]
                dest_filename = f"active_model{file_extension}"
                
                # Get source path
                src_path = model.file.path
                
                # Define FastAPI directory path
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                fastapi_dir = os.path.join(base_dir, 'FastAPI')
                
                # Create the directory if it doesn't exist
                if not os.path.exists(fastapi_dir):
                    os.makedirs(fastapi_dir)
                
                # Set destination path
                dest_path = os.path.join(fastapi_dir, dest_filename)
                
                # Copy the file
                shutil.copy2(src_path, dest_path)

                # Create and save metadata file with requires_scaling information
                metadata = {
                    "name": model.model_name,
                    "description": model.description,
                    "model_type": model.model_type,
                    "requires_scaling": model.requires_scaling,
                    "file_type": file_extension[1:],  # Remove the dot
                    "last_updated": datetime.now().isoformat()
                }
                
                # Save metadata to FastAPI directory
                metadata_path = os.path.join(fastapi_dir, "active_model_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
                
                messages.success(request, f"Model '{model_name}' uploaded and set as active. This model will be used for predictions.")
            except Exception as e:
                messages.warning(request, f"Model uploaded but failed to copy to FastAPI directory: {str(e)}")
        else:
            messages.success(request, f"Model '{model_name}' uploaded successfully.")
        
        return redirect('model_management')
    
    # Auto-register models from filesystem
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'media', 'ml_models')

    # Get all models grouped by type
    models = MLModel.objects.all().order_by('-upload_date')
    active_model = models.filter(is_active=True).first()

    # Get existing model filenames from database
    registered_filenames = set(os.path.basename(model.file.name) for model in models)
    
    # Scan directory for unregistered model files
    registered_count = 0
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith(('.pkl', '.h5')) and filename not in registered_filenames:
                # Extract model name (text before extension)
                model_name = os.path.splitext(filename)[0]
                
                # Create new model record with default metadata
                new_model = MLModel(
                    model_name=model_name,
                    model_type="Unknown",
                    description="Auto-registered from filesystem",
                    file=f"ml_models/{filename}", 
                    uploaded_by=request.user, 
                    is_active=False,
                    requires_scaling=False
                )
                new_model.save()
                registered_count += 1
    
    if registered_count > 0:
        messages.info(request, f"Auto-registered {registered_count} new models from filesystem.")
        # Refresh models list after registration
        models = MLModel.objects.all().order_by('-upload_date')

    # Check file existence for each model in both possible locations
    for model in models:
        try:
            # Path from Django model
            django_path = model.file.path
            
            # Alternative path in FastAPI directory
            filename = os.path.basename(model.file.name)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fastapi_media_path = os.path.join(base_dir, 'FastAPI', 'media', 'ml_models', filename)
            
            # Check both possible locations
            if os.path.exists(django_path) or os.path.exists(fastapi_media_path):
                model.file_exists = True
                # Debug print to see which path exists
                if os.path.exists(django_path):
                    print(f"File found in Django path: {django_path}")
                if os.path.exists(fastapi_media_path):
                    print(f"File found in FastAPI path: {fastapi_media_path}")
            else:
                model.file_exists = False
                print(f"File not found in either location:")
                print(f"Django path (checked): {django_path}")
                print(f"FastAPI path (checked): {fastapi_media_path}")
        except Exception as e:
            print(f"Error checking file existence: {str(e)}")
            model.file_exists = False
    
    return render(request, 'models/model_management.html', {
        'models': models,
        'active_model': active_model
    })

@login_required
def set_model_active(request, model_id):
    """AJAX view to set a model as active"""
    if request.user.role != 'AI Engineer' and request.user.role != 'Admin':
        return JsonResponse({"success": False, "error": "Permission denied"})
    
    try:
        # Get the model to activate
        model = MLModel.objects.get(id=model_id)
        
        # Deactivate all other models (this is handled in the save method)
        model.is_active = True
        model.save()
        
        # Copy model file to FastAPI directory with a generic name
        try:
            # Get file extension and build destination filename
            file_extension = os.path.splitext(model.file.name)[1]
            dest_filename = f"active_model{file_extension}"
            
            # Get source path
            src_path = model.file.path
            
            # Define FastAPI directory path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fastapi_dir = os.path.join(base_dir, 'FastAPI')
            
            # Create the directory if it doesn't exist
            if not os.path.exists(fastapi_dir):
                os.makedirs(fastapi_dir)
            
            # Set destination path
            dest_path = os.path.join(fastapi_dir, dest_filename)
            
            # Copy the file
            shutil.copy2(src_path, dest_path)

            # Create and save metadata file with requires_scaling information
            metadata = {
                "name": model.model_name,
                "description": model.description,
                "model_type": model.model_type,
                "requires_scaling": model.requires_scaling,
                "file_type": file_extension[1:],  # Remove the dot
                "last_updated": datetime.now().isoformat()
            }

            # Save metadata to FastAPI directory
            metadata_path = os.path.join(fastapi_dir, "active_model_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                print(f"Metadata saved to {metadata_path}")
                print(f"Metadata content: {metadata}")

            # After copying the model and metadata, tell FastAPI to reload
            try:
                from utils.ml_api_client import MLApiClient
                
                # Get access token for the request
                client = MLApiClient()
                response = client.reload_models(request=request)
                
                if response.get('success'):
                    print("FastAPI service reloaded model successfully")
                else:
                    print(f"FastAPI service failed to reload model: {response.get('error')}")
                    
            except Exception as reload_error:
                print(f"Error notifying FastAPI service to reload model: {str(reload_error)}")
            
            return JsonResponse({
                "success": True, 
                "message": f"Model {model.model_name} is now active for all predictions"
            })
        except Exception as e:
            return JsonResponse({
                "success": False, 
                "error": f"Failed to copy model file: {str(e)}"
            })
        
    except MLModel.DoesNotExist:
        return JsonResponse({"success": False, "error": "Model not found"})
    except Exception as e:
        error_msg = f"General error: {str(e)}, type: {type(e).__name__}"
        print(error_msg)
        return JsonResponse({"success": False, "error": str(e)})

@login_required
def delete_model(request, model_id):
    """AJAX view to delete a model"""
    if request.user.role != 'AI Engineer' and request.user.role != 'Admin':
        return JsonResponse({"success": False, "error": "Permission denied"})
    
    if request.method != 'POST':
        return JsonResponse({"success": False, "error": "Invalid request method"})
    
    try:
        # Get the model to delete
        model = MLModel.objects.get(id=model_id)
        
        # Don't allow deleting active models
        if model.is_active:
            return JsonResponse({
                "success": False, 
                "error": "Cannot delete an active model. Make another model active first."
            })
        
        # Get the file path before deleting the model
        file_path = model.file.path
        
        # Store name for the response message
        model_name = model.model_name
        
        # Delete the model from the database
        model.delete()
        
        # Remove the file from the filesystem if it exists
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File deleted: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except Exception as file_error:
            print(f"Error deleting file: {str(file_error)}")
            # We still return success since the DB record was deleted
            return JsonResponse({
                "success": True, 
                "message": f"Model {model_name} was deleted from database but the file could not be removed: {str(file_error)}"
            })
        
        return JsonResponse({
            "success": True, 
            "message": f"Model {model_name} deleted successfully"
        })
        
    except MLModel.DoesNotExist:
        return JsonResponse({"success": False, "error": "Model not found"})
    except Exception as e:
        error_msg = f"Error deleting model: {str(e)}"
        print(error_msg)
        return JsonResponse({"success": False, "error": error_msg})

#########################################################################
# ADMIN ANALYTICS VIEWS
#########################################################################
@login_required
def admin_analytics(request):
    """Admin analytics dashboard view"""
    if request.user.role != Role.ADMIN:
        messages.error(request, "Access denied. Admin privileges required.")
        return redirect('dashboard')
    
    # Get date range filter from query params, default to last 30 days
    date_range = request.GET.get('date_range', '30')
    
    if date_range == 'all':
        start_date = None
    else:
        try:
            days = int(date_range)
            start_date = timezone.now() - timedelta(days=days)
        except ValueError:
            start_date = timezone.now() - timedelta(days=30)
    
    # Process date for filtering
    filter_kwargs = {}
    if start_date:
        filter_kwargs = {'timestamp__gte': start_date}
    
    # Get analytics data - Functions are defined below, so we need to move this after the function definitions
    # We'll return the rendered response from get_analytics_data instead
    return get_analytics_data(request, start_date, date_range)

def get_analytics_data(request, start_date, date_range):
    """Get analytics data and render the template"""
    # Get analytics data
    user_metrics = get_user_metrics(start_date)
    #financial_metrics = get_financial_metrics(start_date)
    system_health = get_system_health_metrics(start_date)
    prediction_metrics = get_prediction_metrics(start_date)
    
    context = {
        'user_metrics': user_metrics,
        #'financial_metrics': financial_metrics,
        'system_health': system_health,
        'prediction_metrics': prediction_metrics,
        'date_range': date_range,
    }    
    return render(request, 'analytics/dashboard.html', context)

def get_user_metrics(start_date=None):
    """Generate user engagement metrics"""
    # Filter based on date if provided
    user_filter = {}
    if start_date:
        user_filter = {'member_since__gte': start_date}
    
    prediction_filter = {}
    if start_date:
        prediction_filter = {'timestamp__gte': start_date}
    
    # User registration over time
    if start_date and (timezone.now() - start_date).days < 60:
        # For shorter periods, group by day
        users_over_time = User.objects.filter(**user_filter) \
            .annotate(date=TruncDay('member_since')) \
            .values('date') \
            .annotate(count=Count('id')) \
            .order_by('date')
    else:
        # For longer periods, group by month
        users_over_time = User.objects.filter(**user_filter) \
            .annotate(date=TruncMonth('member_since')) \
            .values('date') \
            .annotate(count=Count('id')) \
            .order_by('date')
    
    # User distribution by role
    role_distribution = User.objects.values('role') \
        .annotate(count=Count('id')) \
        .order_by('role')
    
    # Active users (users who made predictions)
    active_users = Prediction.objects.filter(**prediction_filter) \
        .values('user') \
        .distinct() \
        .count()
    
    # User engagement (predictions per user)
    user_predictions = Prediction.objects.filter(**prediction_filter) \
        .values('user__name', 'user__email') \
        .annotate(prediction_count=Count('id')) \
        .order_by('-prediction_count')[:10]  # Top 10 active users
    
    # Format data for charts
    user_growth_labels = [entry['date'].strftime('%Y-%m-%d') for entry in users_over_time]
    user_growth_data = [entry['count'] for entry in users_over_time]
    user_growth_cumulative = []
    cumulative = 0
    for count in user_growth_data:
        cumulative += count
        user_growth_cumulative.append(cumulative)
    
    role_labels = [entry['role'] for entry in role_distribution]
    role_data = [entry['count'] for entry in role_distribution]
    
    # Package data for the template
    return {
        'total_users': User.objects.count(),
        'new_users': User.objects.filter(**user_filter).count(),
        'active_users': active_users,
        'inactive_users': User.objects.count() - active_users,
        'user_growth_labels': json.dumps(user_growth_labels),
        'user_growth_data': json.dumps(user_growth_cumulative),
        'role_labels': json.dumps(role_labels),
        'role_data': json.dumps(role_data),
        'top_users': user_predictions
    }

def get_system_health_metrics(start_date=None):
    """Generate system health metrics data"""
    # Filter based on date if provided
    metrics_filter = {}
    if start_date:
        metrics_filter = {'timestamp__gte': start_date}
    
    try:
        # API response time data
        api_metrics = APIMetrics.objects.filter(**metrics_filter)
        
        if not api_metrics.exists():
            # If no API metrics, return placeholder data
            return {
                'avg_response_time': 0,
                'error_rate': 0,
                'status_codes': {},
                'response_time_labels': json.dumps([]),
                'response_time_data': json.dumps([])
            }
        
        # Response time over time
        if start_date and (timezone.now() - start_date).days < 7:
            # For shorter periods, group by hour
            response_time_data = api_metrics \
                .annotate(hour=TruncDay('timestamp')) \
                .values('hour') \
                .annotate(avg_time=Avg('response_time')) \
                .order_by('hour')
        else:
            # For longer periods, group by day
            response_time_data = api_metrics \
                .annotate(day=TruncDay('timestamp')) \
                .values('day') \
                .annotate(avg_time=Avg('response_time')) \
                .order_by('day')
        
        # Error rate
        total_requests = api_metrics.count()
        error_requests = api_metrics.filter(error=True).count()
        error_rate = (error_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Status code distribution
        status_codes = api_metrics.values('status_code') \
            .annotate(count=Count('id')) \
            .order_by('status_code')
        
        # Format data for charts
        if start_date and (timezone.now() - start_date).days < 7:
            time_labels = [entry['hour'].strftime('%Y-%m-%d %H:00') for entry in response_time_data]
        else:
            time_labels = [entry['day'].strftime('%Y-%m-%d') for entry in response_time_data]
            
        time_data = [float(entry['avg_time']) for entry in response_time_data]
        
        # Average response time
        avg_response_time = api_metrics.aggregate(Avg('response_time'))['response_time__avg'] or 0
        
        status_summary = {}
        for entry in status_codes:
            status_code = entry['status_code']
            count = entry['count']
            if 200 <= status_code < 300:
                key = '2xx Success'
            elif 300 <= status_code < 400:
                key = '3xx Redirection'
            elif 400 <= status_code < 500:
                key = '4xx Client Error'
            elif 500 <= status_code < 600:
                key = '5xx Server Error'
            else:
                key = 'Other'
                
            if key in status_summary:
                status_summary[key] += count
            else:
                status_summary[key] = count
        
        return {
            'avg_response_time': round(avg_response_time, 2),
            'error_rate': round(error_rate, 2),
            'status_codes': status_summary,
            'status_code_labels': json.dumps(list(status_summary.keys())),
            'status_code_data': json.dumps(list(status_summary.values())),
            'response_time_labels': json.dumps(time_labels),
            'response_time_data': json.dumps(time_data)
        }
    except Exception as e:
        print(f"Error generating system health metrics: {str(e)}")
        return {
            'avg_response_time': 0,
            'error_rate': 0,
            'status_codes': {},
            'response_time_labels': json.dumps([]),
            'response_time_data': json.dumps([])
        }

def get_prediction_metrics(start_date=None):
    """Generate prediction analytics data"""
    # Filter based on date if provided
    prediction_filter = {}
    if start_date:
        prediction_filter = {'timestamp__gte': start_date}
    
    predictions = Prediction.objects.filter(**prediction_filter)
    
    # Prediction volume over time
    if start_date and (timezone.now() - start_date).days < 60:
        # For shorter periods, group by day
        predictions_over_time = predictions \
            .annotate(date=TruncDay('timestamp')) \
            .values('date') \
            .annotate(count=Count('id')) \
            .order_by('date')
    else:
        # For longer periods, group by month
        predictions_over_time = predictions \
            .annotate(date=TruncMonth('timestamp')) \
            .values('date') \
            .annotate(count=Count('id')) \
            .order_by('date')
    
    # Prediction accuracy metrics
    feedback_predictions = predictions.filter(is_satisfactory__isnull=False)
    accurate_predictions = feedback_predictions.filter(is_satisfactory=True).count()
    disputed_predictions = feedback_predictions.filter(is_satisfactory=False).count()
    needs_review = predictions.filter(needs_review=True).count()
    
    total_feedback = accurate_predictions + disputed_predictions
    accuracy = (accurate_predictions / total_feedback * 100) if total_feedback > 0 else 0
    
    # Loan approval analysis
    approved_count = predictions.filter(loan_approval=True).count()
    total_count = predictions.count()
    approval_rate = (approved_count / total_count * 100) if total_count > 0 else 0
    
    # Format data for charts
    prediction_labels = [entry['date'].strftime('%Y-%m-%d') for entry in predictions_over_time]
    prediction_data = [entry['count'] for entry in predictions_over_time]
    
    # Accuracy over time
    if feedback_predictions.exists():
        accuracy_over_time = feedback_predictions \
            .annotate(date=TruncMonth('timestamp')) \
            .values('date') \
            .annotate(
                accurate=Count('id', filter=Q(is_satisfactory=True)),
                total=Count('id')
            ).order_by('date')
        
        accuracy_labels = [entry['date'].strftime('%Y-%m-%d') for entry in accuracy_over_time]
        accuracy_data = [
            round(entry['accurate'] / entry['total'] * 100, 1) if entry['total'] > 0 else 0 
            for entry in accuracy_over_time
        ]
    else:
        accuracy_labels = []
        accuracy_data = []
    
    return {
        'total_predictions': predictions.count(),
        'feedback_provided': total_feedback,
        'accuracy_rate': round(accuracy, 2),
        'disputed_predictions': disputed_predictions,
        'needs_review': needs_review,
        'approval_rate': round(approval_rate, 2),
        'approved_count': approved_count,
        'rejected_count': total_count - approved_count,
        'prediction_labels': json.dumps(prediction_labels),
        'prediction_data': json.dumps(prediction_data),
        'accuracy_labels': json.dumps(accuracy_labels),
        'accuracy_data': json.dumps(accuracy_data)
    }

@login_required
def log_api_metrics(request):
    """API endpoint to log metrics about API performance"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Create new APIMetrics entry
            APIMetrics.objects.create(
                endpoint=data.get('endpoint', 'unknown'),
                response_time=float(data.get('response_time', 0)),
                status_code=int(data.get('status_code', 500)),
                error=bool(data.get('error', False))
            )
            
            return JsonResponse({"success": True})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    
    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)

@login_required
def export_analytics_data(request):
    """Export analytics data as CSV for admins"""
    if request.user.role != Role.ADMIN:
        messages.error(request, "Access denied. Admin privileges required.")
        return redirect('dashboard')
    
    import csv
    
    # Create the HttpResponse object with CSV header
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="analytics_export.csv"'
    
    # Create CSV writer
    writer = csv.writer(response)
    
    # Determine the data type to export
    export_type = request.GET.get('type', 'user')
    
    if export_type == 'user':
        # User analytics export
        writer.writerow(['User ID', 'Email', 'Name', 'Role', 'Member Since', 'Predictions Made'])
        
        users = User.objects.all().order_by('id')
        for user in users:
            writer.writerow([
                user.id, 
                user.email, 
                user.name, 
                user.role, 
                user.member_since.strftime('%Y-%m-%d'),
                Prediction.objects.filter(user=user).count()
            ])
   
    elif export_type == 'prediction':
        # Prediction analytics export
        writer.writerow(['Prediction ID', 'User', 'Timestamp', 'Loan Approval', 
                        'Confidence Score', 'Is Satisfactory', 'Needs Review', 'Is Checked'])
        
        predictions = Prediction.objects.all().order_by('-timestamp')
        for prediction in predictions:
            writer.writerow([
                prediction.id,
                prediction.user.email,
                prediction.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Approved' if prediction.loan_approval else 'Rejected',
                float(prediction.confidence_score) * 100 if hasattr(prediction, 'confidence_score') else 0,
                prediction.is_satisfactory if prediction.is_satisfactory is not None else 'No Feedback',
                prediction.needs_review,
                prediction.is_checked
            ])
            
    elif export_type == 'system':
        # System health analytics export
        writer.writerow(['Metric ID', 'Endpoint', 'Response Time (ms)', 'Timestamp', 
                         'Status Code', 'Error'])
        
        metrics = APIMetrics.objects.all().order_by('-timestamp')
        for metric in metrics:
            writer.writerow([
                metric.id,
                metric.endpoint,
                metric.response_time,
                metric.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                metric.status_code,
                metric.error
            ])
    
    return response
