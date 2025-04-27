from djoser.serializers import UserCreateSerializer, UserSerializer as DjoserUserSerializer
from rest_framework import serializers
from .models import User, Prediction

# User Serializer using Djoser
class CustomUserSerializer(DjoserUserSerializer):
    class Meta:
        model = User
        fields = ['email', 'name', 'role', 'member_since']

# Custom Registration Serializer
class CustomUserCreateSerializer(UserCreateSerializer):
    class Meta(UserCreateSerializer.Meta):
        model = User
        fields = ['email', 'name', 'password']

"""
# Invoice Serializer
class InvoiceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Invoice
        fields = ['id', 'user', 'amount', 'issued_date', 'due_date', 'status']"
"""

# Prediction Serializer
class PredictionSerializer(serializers.ModelSerializer):
    user_name = serializers.SerializerMethodField()
    
    class Meta:
        model = Prediction
        fields = ['prediction_id', 'user', 'timestamp', 'input_data', 'result', 'settlement_value', 
                 'is_satisfactory', 'adjustment_rationale', 'needs_review', 
                 'feedback_date', 'is_checked']
    
    def get_user_name(self, obj):
        return obj.user.name if obj.user else None
    
    