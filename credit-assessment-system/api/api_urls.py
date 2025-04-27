## All the endpoints for the API For DRF
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .viewsets import UserViewSet, PredictionManagementViewSet

# Set up the router and register viewsets
router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'predictions', PredictionManagementViewSet)

# API URL patterns
urlpatterns = [
    path('', include(router.urls)),
    # Add any other custom API endpoints here
]