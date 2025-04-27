from datetime import datetime, timedelta
from typing import Optional,Union

from fastapi import Depends, HTTPException, status, Request 
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel

## Django Production Secret Key (iykyk)
SECRET_KEY = 'django-insecure-hw4vd2#ltj2ww^t2@y2qett$7(!+t-021^6)5e6wqp!uzt=m2='
ALGORITHM = "HS256"  
ACCESS_TOKEN_EXPIRE_MINUTES = 60  

## Extracting token from authorization header 
security = HTTPBearer()

class TokenData(BaseModel):
    access_token: str
    token_type: str

## Implement Few Ways to obtain JWT Token 
# 1. From the Web Cookies
# 2. From the Auth Headers

async def get_token_from_header(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Union[str, None]:
    """Extract token from the Authorization header"""
    if credentials:
        return credentials.credentials
    return None

async def get_token_from_cookie(request: Request) -> Union[str, None]:
    """Extract token from cookies"""
    return request.cookies.get("access_token")

async def get_current_user(
    request: Request,
    header_token: Optional[str] = Depends(get_token_from_header)
) -> dict:
    """Validate JWT token and return user information"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # First try header token, then cookie
    token = header_token
    if not token:
        token = await get_token_from_cookie(request)
    
    if not token:
        raise credentials_exception
    
    try:
        # Decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Get user information from payload
        user_id = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
        
        # Return user information
        return payload
    except JWTError:
        raise credentials_exception