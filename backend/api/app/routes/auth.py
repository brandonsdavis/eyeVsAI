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

from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import httpx
import secrets
from urllib.parse import urlencode

from ..database import get_db
from ..auth import auth_manager, get_current_user, get_current_user_optional, OAUTH_CONFIGS
from ..models import (
    UserRegistration, UserLogin, TokenResponse, RefreshTokenRequest,
    UserProfile, UserProfileUpdate, OAuthProvider, OAuthAuthorizationURL, OAuthCallback
)
from ..db_models import User, RefreshToken

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=TokenResponse)
async def register(
    user_data: UserRegistration,
    request: Request,
    db: Session = Depends(get_db)
):
    """Register a new user with email and password."""
    try:
        user = auth_manager.create_user(
            db=db,
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.display_name
        )
        
        # Log registration event
        auth_manager.log_event(
            db=db,
            user_id=str(user.id),
            event_type="user_registration",
            event_data={"email": user_data.email},
            request=request
        )
        
        # Create tokens
        access_token = auth_manager.create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        refresh_token = auth_manager.create_refresh_token(
            user_id=str(user.id),
            db=db,
            request=request
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=30 * 60  # 30 minutes
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    user_data: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """Login with email and password."""
    user = auth_manager.authenticate_user(db, user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Log login event
    auth_manager.log_event(
        db=db,
        user_id=str(user.id),
        event_type="user_login",
        event_data={"email": user_data.email},
        request=request
    )
    
    # Create tokens
    access_token = auth_manager.create_access_token(
        data={"sub": str(user.id), "email": user.email}
    )
    refresh_token = auth_manager.create_refresh_token(
        user_id=str(user.id),
        db=db,
        request=request
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=30 * 60  # 30 minutes
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token."""
    refresh_token = db.query(RefreshToken).filter(
        RefreshToken.token == refresh_data.refresh_token,
        RefreshToken.is_revoked == False,
        RefreshToken.expires_at > datetime.utcnow()
    ).first()
    
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user = db.query(User).filter(User.id == refresh_token.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new access token
    access_token = auth_manager.create_access_token(
        data={"sub": str(user.id), "email": user.email}
    )
    
    # Optionally rotate refresh token
    new_refresh_token = auth_manager.create_refresh_token(
        user_id=str(user.id),
        db=db,
        request=request
    )
    
    # Revoke old refresh token
    refresh_token.is_revoked = True
    db.commit()
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=30 * 60  # 30 minutes
    )


@router.post("/logout")
async def logout(
    refresh_data: RefreshTokenRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout and revoke refresh token."""
    auth_manager.revoke_refresh_token(db, refresh_data.refresh_token)
    
    # Log logout event
    auth_manager.log_event(
        db=db,
        user_id=str(current_user.id),
        event_type="user_logout",
        request=request
    )
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Get current user profile."""
    return UserProfile.model_validate(current_user)


@router.put("/me", response_model=UserProfile)
async def update_user_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user profile."""
    if profile_data.display_name is not None:
        current_user.display_name = profile_data.display_name
    
    if profile_data.username is not None:
        # Check if username is already taken
        existing_user = db.query(User).filter(
            User.username == profile_data.username,
            User.id != current_user.id
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        current_user.username = profile_data.username
    
    db.commit()
    db.refresh(current_user)
    
    return UserProfile.model_validate(current_user)


# OAuth Routes
@router.get("/oauth/{provider}/authorize", response_model=OAuthAuthorizationURL)
async def oauth_authorize(provider: OAuthProvider):
    """Get OAuth authorization URL."""
    if provider not in OAUTH_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth provider '{provider}' not supported"
        )
    
    config = OAUTH_CONFIGS[provider]
    if not config["client_id"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth provider '{provider}' not configured"
        )
    
    state = secrets.token_urlsafe(32)
    
    # Build authorization URL
    params = {
        "client_id": config["client_id"],
        "redirect_uri": f"http://localhost:3000/auth/callback/{provider}",  # Frontend handles callback
        "response_type": "code",
        "state": state,
        "scope": "openid profile email" if provider == "google" else "read:user user:email"
    }
    
    authorization_url = f"{config['auth_url']}?{urlencode(params)}"
    
    return OAuthAuthorizationURL(
        authorization_url=authorization_url,
        state=state
    )


@router.post("/oauth/callback", response_model=TokenResponse)
async def oauth_callback(
    callback_data: OAuthCallback,
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle OAuth callback and create/login user."""
    if callback_data.provider not in OAUTH_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth provider '{callback_data.provider}' not supported"
        )
    
    config = OAUTH_CONFIGS[callback_data.provider]
    
    try:
        # Exchange code for access token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                config["token_url"],
                data={
                    "client_id": config["client_id"],
                    "client_secret": config["client_secret"],
                    "code": callback_data.code,
                    "grant_type": "authorization_code",
                    "redirect_uri": f"http://localhost:3000/auth/callback/{callback_data.provider}"
                },
                headers={"Accept": "application/json"}
            )
            
            if token_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange code for token"
                )
            
            token_data = token_response.json()
            access_token = token_data.get("access_token")
            
            if not access_token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No access token received"
                )
            
            # Get user info from provider
            if config["userinfo_url"]:
                user_response = await client.get(
                    config["userinfo_url"],
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if user_response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to get user info"
                    )
                
                user_data = user_response.json()
            else:
                # Apple provides user info in token response
                user_data = token_data.get("user", {})
            
            # Normalize user data across providers
            normalized_user_data = normalize_oauth_user_data(callback_data.provider, user_data)
            
            # Create or update user
            user = auth_manager.create_or_update_oauth_user(
                db=db,
                provider=callback_data.provider,
                oauth_user_data=normalized_user_data
            )
            
            # Log OAuth login event
            auth_manager.log_event(
                db=db,
                user_id=str(user.id),
                event_type="oauth_login",
                event_data={"provider": callback_data.provider},
                request=request
            )
            
            # Create tokens
            access_token = auth_manager.create_access_token(
                data={"sub": str(user.id), "email": user.email}
            )
            refresh_token = auth_manager.create_refresh_token(
                user_id=str(user.id),
                db=db,
                request=request
            )
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=30 * 60  # 30 minutes
            )
            
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth request failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth callback failed: {str(e)}"
        )


@router.post("/guest", response_model=TokenResponse)
async def create_guest_session(
    request: Request,
    db: Session = Depends(get_db)
):
    """Create a guest user session."""
    session_id = secrets.token_urlsafe(16)
    
    user = auth_manager.create_guest_user(db, session_id)
    
    # Log guest session creation
    auth_manager.log_event(
        db=db,
        user_id=str(user.id),
        event_type="guest_session_created",
        request=request
    )
    
    # Create tokens
    access_token = auth_manager.create_access_token(
        data={"sub": str(user.id), "guest": True}
    )
    refresh_token = auth_manager.create_refresh_token(
        user_id=str(user.id),
        db=db,
        request=request
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=30 * 60  # 30 minutes
    )


def normalize_oauth_user_data(provider: str, user_data: dict) -> dict:
    """Normalize user data from different OAuth providers."""
    if provider == "google":
        return {
            "id": user_data.get("id"),
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "username": user_data.get("email", "").split("@")[0],
            "avatar_url": user_data.get("picture")
        }
    elif provider == "facebook":
        return {
            "id": user_data.get("id"),
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "username": user_data.get("email", "").split("@")[0],
            "avatar_url": user_data.get("picture", {}).get("data", {}).get("url")
        }
    elif provider == "github":
        return {
            "id": user_data.get("id"),
            "email": user_data.get("email"),
            "name": user_data.get("name") or user_data.get("login"),
            "username": user_data.get("login"),
            "avatar_url": user_data.get("avatar_url")
        }
    elif provider == "discord":
        return {
            "id": user_data.get("id"),
            "email": user_data.get("email"),
            "name": user_data.get("global_name") or user_data.get("username"),
            "username": user_data.get("username"),
            "avatar_url": f"https://cdn.discordapp.com/avatars/{user_data.get('id')}/{user_data.get('avatar')}.png" if user_data.get('avatar') else None
        }
    elif provider == "twitter":
        return {
            "id": user_data.get("id"),
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "username": user_data.get("username"),
            "avatar_url": user_data.get("profile_image_url")
        }
    elif provider == "apple":
        return {
            "id": user_data.get("sub"),
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "username": user_data.get("email", "").split("@")[0],
            "avatar_url": None
        }
    else:
        return user_data