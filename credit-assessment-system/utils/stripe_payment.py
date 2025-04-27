import stripe
from django.conf import settings
from django.urls import reverse
from api.models import Invoice 

## Private Stripe API Key
stripe.api_key = settings.STRIPE_SECRET_KEY

## Create a Stripe Payment Intent for the Invoice 
def create_intent(Invoice):
    try:    
        amount = int(Invoice.amount * 100)
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='gbp',
            metadata={'invoice_id' : Invoice.id ,
                      'invoice_user_email' : Invoice.user.email
                      },
            description=f'Invoice {Invoice.id} for {Invoice.user.email}',
            automatic_payment_methods= True 
        )
        return {
            'client_secret': intent.client_secret,
            'payment_intent_id': intent.id
        }
    except Exception as Exp:
        print(f"Error Occured When Creating Payment Intent: str{Exp}")
        return None
    
def create_checkout(invoice, domain_url):
    """Create a Stripe checkout session for the invoice."""
    try:
        # Add debugging
        print(f"Creating checkout for invoice #{invoice.id} with amount {invoice.amount}")
        
        # Get description 
        description = invoice.description or f'Invoice #{invoice.id}'
        
        # Create line items for the invoice
        line_items = [{
            'price_data': {
                'currency': 'gbp',
                'product_data': {
                    'name': f'Invoice #{invoice.id}',
                    'description': description,
                },
                'unit_amount': int(float(invoice.amount) * 100),  # Amount in pence
            },
            'quantity': 1,
        }]
        
        # Create the checkout session
        print(f"Creating Stripe checkout with domain URL: {domain_url}")
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=line_items,
            mode='payment',
            success_url=f"{domain_url}/user/payment/success/?session_id={{CHECKOUT_SESSION_ID}}&invoice_id={invoice.id}",
            cancel_url=f"{domain_url}/user/payment/cancel/?invoice_id={invoice.id}",
            metadata={
                'invoice_id': str(invoice.id)  # Convert to string for safety
            }
        )
        
        # Print debug information about the session
        print(f"Checkout session created: {checkout_session.id}")
        print(f"Session URL: {checkout_session.url}")
        print(f"Session Payment Intent: {checkout_session.payment_intent}")
        
        return checkout_session
    
    except Exception as exp:
        print(f"Error creating checkout session: {str(exp)}")
        print(f"Invoice details: ID={invoice.id}, Amount={invoice.amount}")
        if hasattr(exp, 'json_body'):
            print(f"Stripe error details: {exp.json_body}")
        return None
    
## Verify the Payment Intent Status 
def verify_intent(payment_intent_id):
    try:
        intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        return intent.status
    except Exception as exp:
        print(f"Error Occured When Trying to Verify Payment Intent: str{exp}")
        return None
