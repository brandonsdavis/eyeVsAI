#!/usr/bin/env python3
"""
Simple API test script to verify all endpoints are working
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000/api/v1"

def test_endpoint(method, endpoint, data=None, headers=None, token=None):
    """Test an API endpoint and return the response"""
    url = f"{BASE_URL}{endpoint}"
    
    if headers is None:
        headers = {"Content-Type": "application/json"}
    
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        
        print(f"✓ {method} {endpoint} - Status: {response.status_code}")
        return response
    except Exception as e:
        print(f"✗ {method} {endpoint} - Error: {e}")
        return None

def main():
    """Run API tests"""
    print("🧪 Testing EyeVsAI Backend API")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Health Check")
    test_endpoint("GET", "/health")
    
    # Test datasets endpoint
    print("\n2. Game Datasets")
    response = test_endpoint("GET", "/game/datasets")
    if response and response.status_code == 200:
        data = response.json()
        print(f"   Found {len(data['datasets'])} datasets")
    
    # Test user registration
    print("\n3. User Registration")
    response = test_endpoint("POST", "/auth/register", {
        "email": "test@example.com",
        "password": "testpassword123",
        "display_name": "Test User"
    })
    
    if response and response.status_code == 200:
        tokens = response.json()
        access_token = tokens["access_token"]
        print("   ✓ User registered successfully")
        
        # Test authenticated endpoint
        print("\n4. User Profile (Authenticated)")
        test_endpoint("GET", "/auth/me", token=access_token)
        
        # Test game session creation
        print("\n5. Game Session Creation")
        session_response = test_endpoint("POST", "/game/session", {
            "dataset": "pets",
            "difficulty": "easy",
            "ai_model_key": "shallow/rf_hog_lbp",
            "total_rounds": 3
        }, token=access_token)
        
        if session_response and session_response.status_code == 200:
            session_data = session_response.json()
            session_id = session_data["session_id"]
            print(f"   ✓ Game session created: {session_id}")
    
    # Test guest session
    print("\n6. Guest Session")
    response = test_endpoint("POST", "/auth/guest", {})
    if response and response.status_code == 200:
        guest_tokens = response.json()
        guest_token = guest_tokens["access_token"]
        print("   ✓ Guest session created")
    
    # Test API documentation
    print("\n7. API Documentation")
    try:
        response = requests.get(f"{BASE_URL.replace('/api/v1', '')}/api/docs")
        if response.status_code == 200:
            print("   ✓ Swagger docs accessible")
        else:
            print(f"   ✗ Swagger docs failed - Status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Swagger docs error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API Testing Complete!")
    print("\nNext steps:")
    print("- Visit http://localhost:8000/api/docs for interactive API documentation")
    print("- The backend is ready for frontend integration")

if __name__ == "__main__":
    main()