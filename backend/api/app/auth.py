# Copyright 2025 Brandon Davis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .database import get_db
from .db_models import User, RefreshToken, AuditLog

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# OAuth Configuration
OAUTH_CONFIGS = {
    "google": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "auth_url": "https://accounts.google.com/o/oauth2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
    },
    "facebook": {
        "client_id": os.getenv("FACEBOOK_CLIENT_ID"),
        "client_secret": os.getenv("FACEBOOK_CLIENT_SECRET"),
        "auth_url": "https://www.facebook.com/v18.0/dialog/oauth",
        "token_url": "https://graph.facebook.com/v18.0/oauth/access_token",
        "userinfo_url": "https://graph.facebook.com/v18.0/me",
    },
    "github": {
        "client_id": os.getenv("GITHUB_CLIENT_ID"),
        "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
        "auth_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
    },
    "discord": {
        "client_id": os.getenv("DISCORD_CLIENT_ID"),
        "client_secret": os.getenv("DISCORD_CLIENT_SECRET"),
        "auth_url": "https://discord.com/oauth2/authorize",
        "token_url": "https://discord.com/api/oauth2/token",
        "userinfo_url": "https://discord.com/api/users/@me",
    },
    "twitter": {
        "client_id": os.getenv("TWITTER_CLIENT_ID"),
        "client_secret": os.getenv("TWITTER_CLIENT_SECRET"),
        "auth_url": "https://twitter.com/i/oauth2/authorize",
        "token_url": "https://api.twitter.com/2/oauth2/token",
        "userinfo_url": "https://api.twitter.com/2/users/me",
    },
    "apple": {
        "client_id": os.getenv("APPLE_CLIENT_ID"),
        "client_secret": os.getenv("APPLE_CLIENT_SECRET"),
        "auth_url": "https://appleid.apple.com/auth/authorize",
        "token_url": "https://appleid.apple.com/auth/token",
        "userinfo_url": None,  # Apple provides user info in the token response
    }
}

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class AuthManager:
    """Handles authentication and authorization."""
    
    def __init__(self):
        self.pwd_context = pwd_context
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str, db: Session, request: Request) -> str:
        """Create refresh token and store in database."""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        refresh_token = RefreshToken(
            user_id=user_id,
            token=token,
            expires_at=expires_at,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None
        )
        
        db.add(refresh_token)
        db.commit()
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.PyJWTError:
            return None
    
    def authenticate_user(self, db: Session, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.password_hash:
            return None
        
        if not self.verify_password(password, user.password_hash):
            return None
        
        return user
    
    def create_user(self, db: Session, email: str, password: str, display_name: str = None) -> User:
        """Create new user with email and password."""
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        user = User(
            email=email,
            password_hash=self.get_password_hash(password),
            display_name=display_name or email.split('@')[0],
            username=email.split('@')[0],
            is_active=True
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return user
    
    def create_or_update_oauth_user(self, db: Session, provider: str, 
                                   oauth_user_data: Dict[str, Any]) -> User:
        """Create or update user from OAuth provider data."""
        provider_id_field = f"{provider}_id"
        oauth_id = str(oauth_user_data.get("id"))
        
        if not oauth_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid OAuth user data"
            )
        
        # Check if user exists by OAuth ID
        user = db.query(User).filter(
            getattr(User, provider_id_field) == oauth_id
        ).first()
        
        if user:
            # Update existing user
            user.display_name = oauth_user_data.get("name", user.display_name)
            user.avatar_url = oauth_user_data.get("avatar_url", user.avatar_url)
            user.last_login = datetime.utcnow()
            db.commit()
            return user
        
        # Check if user exists by email
        email = oauth_user_data.get("email")
        if email:
            user = db.query(User).filter(User.email == email).first()
            if user:
                # Link OAuth account to existing user
                setattr(user, provider_id_field, oauth_id)
                user.display_name = oauth_user_data.get("name", user.display_name)
                user.avatar_url = oauth_user_data.get("avatar_url", user.avatar_url)
                user.last_login = datetime.utcnow()
                db.commit()
                return user
        
        # Create new user
        username = oauth_user_data.get("username") or oauth_user_data.get("login")
        if not username and email:
            username = email.split('@')[0]
        
        user = User(
            email=email,
            username=username,
            display_name=oauth_user_data.get("name", username),
            avatar_url=oauth_user_data.get("avatar_url"),
            is_active=True,
            last_login=datetime.utcnow()
        )
        
        setattr(user, provider_id_field, oauth_id)
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return user
    
    def create_guest_user(self, db: Session, session_id: str) -> User:
        """Create a guest user for temporary sessions."""
        user = User(
            username=f"guest_{session_id[:8]}",
            display_name=f"Guest {session_id[:8]}",
            is_guest=True,
            is_active=True
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return user
    
    def revoke_refresh_token(self, db: Session, token: str) -> bool:
        """Revoke a refresh token."""
        refresh_token = db.query(RefreshToken).filter(
            RefreshToken.token == token
        ).first()
        
        if refresh_token:
            refresh_token.is_revoked = True
            db.commit()
            return True
        
        return False
    
    def log_event(self, db: Session, user_id: str, event_type: str, 
                  event_data: Dict[str, Any] = None, request: Request = None):
        """Log audit event."""
        audit_log = AuditLog(
            user_id=user_id,
            event_type=event_type,
            event_data=event_data,
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("user-agent") if request else None
        )
        
        db.add(audit_log)
        db.commit()


auth_manager = AuthManager()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, otherwise None."""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None