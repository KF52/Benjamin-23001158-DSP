## This file is dedicated to the middleware functionality for the Cookies carrying the User Auth Token 
from django.utils.functional import SimpleLazyObject
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.contrib.auth.models import AnonymousUser

def get_user_from_token(request):
    ## Get the user from the JWT token in the cookie.
    token = request.COOKIES.get('access_token')
    if not token:
        return AnonymousUser()
    
    ## Add the token to the Authorization header
    request.META['HTTP_AUTHORIZATION'] = f"Bearer {token}"
    
    auth = JWTAuthentication()
    try:
        validated_token = auth.get_validated_token(token)
        user = auth.get_user(validated_token)
        return user
    except Exception:
        return AnonymousUser()

class JWTCookieMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.user = SimpleLazyObject(lambda: get_user_from_token(request))
        return self.get_response(request)