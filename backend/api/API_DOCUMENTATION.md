# EyeVsAI Game API Documentation

## Overview

The EyeVsAI Game API provides endpoints for user authentication, game management, leaderboards, and AI model interactions.

Base URL: `http://localhost:8000/api/v1`

## Authentication

Most endpoints require authentication via JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### Authentication

#### Register New User
- **POST** `/auth/register`
- **Body**: `{ "email": "user@example.com", "password": "password123", "display_name": "User Name" }`
- **Response**: `{ "access_token": "...", "refresh_token": "...", "token_type": "bearer", "expires_in": 1800 }`

#### Login
- **POST** `/auth/login`
- **Body**: `{ "email": "user@example.com", "password": "password123" }`
- **Response**: Same as register

#### OAuth Login
- **GET** `/auth/oauth/{provider}/authorize`
- **Providers**: google, facebook, github, discord, twitter, apple
- **Response**: `{ "authorization_url": "...", "state": "..." }`

#### OAuth Callback
- **POST** `/auth/oauth/callback`
- **Body**: `{ "code": "...", "state": "...", "provider": "google" }`
- **Response**: Same as register

#### Guest Session
- **POST** `/auth/guest`
- **Response**: Same as register (temporary account)

#### Refresh Token
- **POST** `/auth/refresh`
- **Body**: `{ "refresh_token": "..." }`
- **Response**: New access and refresh tokens

#### Get Profile
- **GET** `/auth/me` (Requires auth)
- **Response**: User profile information

#### Update Profile
- **PUT** `/auth/me` (Requires auth)
- **Body**: `{ "display_name": "New Name", "username": "newusername" }`
- **Response**: Updated user profile

### Game Management

#### Get Available Datasets
- **GET** `/game/datasets`
- **Response**: List of available datasets with difficulty levels and models

#### Create Game Session
- **POST** `/game/session`
- **Body**: `{ "dataset": "pets", "difficulty": "medium", "ai_model_key": "transfer/resnet50", "total_rounds": 10 }`
- **Response**: Game session information

#### Get Game Session
- **GET** `/game/session/{session_id}`
- **Response**: Current game session state

#### Create Game Round
- **POST** `/game/session/{session_id}/round`
- **Response**: `{ "round_id": "...", "round_number": 1, "image_url": "...", "options": ["cat", "dog", "bird", "fish"], "ai_commitment_hash": "..." }`

#### Submit Round Answer
- **POST** `/game/round/{round_id}/submit`
- **Body**: `{ "round_id": "...", "user_answer": "cat", "response_time_ms": 5000 }`
- **Response**: Round result with AI prediction reveal

#### Complete Game Session
- **POST** `/game/session/{session_id}/complete`
- **Response**: Final game summary with scores

### Leaderboards

#### Get Leaderboard
- **GET** `/game/leaderboard/{dataset}/{difficulty}?period=all_time&limit=100`
- **Periods**: daily, weekly, monthly, all_time
- **Response**: Leaderboard entries with user rankings

#### Get User Stats
- **GET** `/game/user/stats` (Requires auth)
- **Response**: User's game statistics and history

### Images

#### Serve Game Images
- **GET** `/images/{image_path}`
- **Response**: Image file for game rounds

### Legacy Endpoints (for backward compatibility)

#### Predict Image
- **POST** `/predict`
- **Body**: `{ "image_data": "<base64>", "model_type": "transfer", "top_k": 3 }`
- **Response**: Model predictions

#### List Models
- **GET** `/models`
- **Response**: Available models information

## Error Responses

All errors follow this format:
```json
{
  "detail": "Error message"
}
```

Common status codes:
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

## Rate Limiting

API rate limits (when enabled):
- Authenticated users: 1000 requests/hour
- Guest users: 100 requests/hour
- OAuth endpoints: 10 requests/minute

## WebSocket Support (Coming Soon)

Real-time updates will be available at:
- `ws://localhost:8000/ws`

Events:
- Game round updates
- Leaderboard changes
- Live player counts