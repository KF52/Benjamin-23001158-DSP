from rest_framework import viewsets, permissions
from .models import User, Role, Prediction
from .permissions import IsAdminUser, IsStaff, IsAIEngineer
from rest_framework.response import Response
from .serializers import CustomUserSerializer, PredictionSerializer
from rest_framework.decorators import action
from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiParameter, OpenApiExample

@extend_schema_view(
    list=extend_schema(
        summary="List all users",
        description="Get a list of all users. Requires admin access."
    ),
    retrieve=extend_schema(
        summary="Retrieve user details",
        description="Get details for a specific user."
    ),
    update=extend_schema(
        summary="Update user",
        description="Update an existing user's information. Requires admin access."
    ),
    partial_update=extend_schema(
        summary="Partially update user",
        description="Update part of an existing user's information. Requires admin access."
    ),
    destroy=extend_schema(
        summary="Delete user",
        description="Delete a user. Requires admin access."
    )
)
class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing users.
    
    This viewset provides CRUD operations for User objects.
    Only administrators can assign roles to users.
    """
    queryset = User.objects.all()
    serializer_class = CustomUserSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_permissions(self):
        if self.action in ['assign_role']:
            return [permissions.IsAdminUser()]
        return super().get_permissions()
    
    @extend_schema(
        summary="Update user role",
        description="Change a user's role. Only admins can update roles.",
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'role': {'type': 'string', 'enum': [choice[0] for choice in Role.choices]}
                },
                'required': ['role']
            }
        },
        responses={
            200: {'description': 'Role updated successfully'},
            400: {'description': 'Invalid role provided'},
            403: {'description': 'Not enough permissions'}
        }
    )
    @action(detail=True, methods=['post'])
    def update_role(self, request, pk=None):
        """
        Update a user's role.
        
        Only administrators can change user roles.
        Valid roles are: AI Engineer, FinanceTeam, and Admin.
        """
        user = self.get_object()
        if request.user.role != Role.ADMIN:
            return Response({'error': 'Only Admins can Alter roles'}, status=403)
        
        role = request.data.get('role')
        if role not in [choice[0] for choice in Role.choices]:
            return Response({'error': 'Invalid role, Select Either AI Engineer, FinanceTeam or Admin'}, status=400)
        
        user.role = role
        user.save()
        return Response({'message': f'Role updated to {role}'}, status=200)

# Add a new PredictionManagementViewSet for Staff
@extend_schema_view(
    list=extend_schema(
        summary="List all predictions",
        description="Get a list of all predictions. Requires staff or admin access."
    ),
    retrieve=extend_schema(
        summary="Retrieve prediction details",
        description="Get details for a specific prediction."
    ),
)
class PredictionManagementViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for managing predictions.
    
    This viewset provides read operations for Prediction objects.
    Staff and admin users can review prediction history.
    """
    queryset = Prediction.objects.all().order_by('-timestamp')
    serializer_class = PredictionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return [IsStaff() | IsAdminUser()]
        return super().get_permissions()
    
    @extend_schema(
        summary="Review prediction",
        description="Mark a prediction as reviewed by staff.",
        responses={
            200: {'description': 'Prediction reviewed successfully'},
            403: {'description': 'Not enough permissions'}
        }
    )
    @action(detail=True, methods=['post'])
    def review(self, request, pk=None):
        """
        Mark a prediction as reviewed.
        
        Only staff and admins can review predictions.
        """
        prediction = self.get_object()
        if request.user.role not in [Role.STAFF, Role.ADMIN]:
            return Response({'error': 'Only Staff and Admins can review predictions'}, status=403)
        
        prediction.is_checked = True
        prediction.save()
        return Response({'message': f'Prediction marked as reviewed'}, status=200)

"""
@extend_schema_view(
    list=extend_schema(
        summary="List all invoices",
        description="Get a list of all invoices. Requires finance team or admin access."
    ),
    retrieve=extend_schema(
        summary="Retrieve invoice details",
        description="Get details for a specific invoice."
    ),
    create=extend_schema(
        summary="Create invoice",
        description="Create a new invoice. Requires finance team or admin access."
    ),
    update=extend_schema(
        summary="Update invoice",
        description="Update an existing invoice. Requires finance team or admin access."
    ),
    partial_update=extend_schema(
        summary="Partially update invoice",
        description="Update part of an existing invoice. Requires finance team or admin access."
    ),
    destroy=extend_schema(
        summary="Delete invoice",
        description="Delete an invoice. Requires finance team or admin access."
    )
)

class InvoiceViewSet(viewsets.ModelViewSet):
    
    API endpoint for managing invoices.
    
    This viewset provides CRUD operations for Invoice objects.
    Finance team and admin users can create, update, and delete invoices.
    
    queryset = Invoice.objects.all()
    serializer_class = InvoiceSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_permissions(self):
        if self.action in ['view', 'retrieve', 'create', 'update', 'partial_update', 'destroy']:
            return [IsFinanceTeam() | IsAdminUser()]
        return super().get_permissions()
    
    def perform_create(self, serializer):
        user_id = self.request.data.get('user')
        user = User.objects.get(id=user_id)
        serializer.save(user=user)
    
    @extend_schema(
        summary="Generate payment link",
        description="Create a payment link for an invoice using Stripe.",
        responses={
            200: {
                'description': 'Payment link generated successfully',
                'type': 'object',
                'properties': {
                    'payment_url': {'type': 'string', 'format': 'uri'}
                }
            },
            400: {'description': 'Could not create payment session'}
        }
    )
    @action(detail=True, methods=['post'])
    def payment_link(self, request, pk=None):
        
        Generate a Stripe payment link for the invoice.
        
        Creates a Stripe Checkout session and returns the payment URL.
        Updates the invoice with payment details.
        
        invoice = self.get_object()
        domain_url = request.build_absolute_uri('/').rstrip('/')

        checkout_session = create_checkout(invoice, domain_url)

        if checkout_session:
            # Update the invoice with payment details
            invoice.payment_url = checkout_session.url
            invoice.stripe_payment_intent_id = checkout_session.payment_intent
            invoice.save()

            return Response({
                'payment_url': checkout_session.url
            })
        else:
            return Response({
                'error': 'Could not create payment session'
            }, status=400)
    
    @extend_schema(
        summary="Verify payment status",
        description="Check the payment status of an invoice with Stripe.",
        responses={
            200: {
                'description': 'Payment status',
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'invoice_status': {'type': 'string'}
                }
            }
        }
    )
    @action(detail=True, methods=['get'])
    def verify_payment(self, request, pk=None):
        
        Check the payment status of an invoice.
        
        Verifies the payment status with Stripe and updates the invoice status
        if the payment has been successfully processed.
        
        invoice = self.get_object()
        
        if not invoice.stripe_payment_intent_id:
            return Response({
                'status': 'No payment initiated'
            })
        
        payment_status = verify_payment_intent(invoice.stripe_payment_intent_id)
        
        # Update invoice status if payment is successful
        if payment_status == 'succeeded' and invoice.status == 'Pending':
            invoice.status = 'Paid'
            invoice.save()
        
        return Response({
            'status': payment_status,
            'invoice_status': invoice.status
        })
"""