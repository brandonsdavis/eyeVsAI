import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import './AuthPage.css';

const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    displayName: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const { login, isAuthenticated } = useAuth();

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/game');
    }
  }, [isAuthenticated, navigate]);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    // Clear error when user starts typing
    if (error) setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const endpoint = isLogin ? '/api/v1/auth/login' : '/api/v1/auth/register';
      const body = isLogin 
        ? { email: formData.email, password: formData.password }
        : { 
            email: formData.email, 
            password: formData.password, 
            display_name: formData.displayName 
          };

      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (response.ok) {
        // Use the auth context to handle login
        login(data.access_token, data.refresh_token);
        
        // Redirect to game
        navigate('/game');
      } else {
        setError(data.detail || 'Authentication failed');
      }
    } catch (err) {
      setError('Network error. Please check if the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleGuestLogin = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/api/v1/auth/guest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });

      const data = await response.json();

      if (response.ok) {
        login(data.access_token, data.refresh_token);
        navigate('/game');
      } else {
        setError(data.detail || 'Guest login failed');
      }
    } catch (err) {
      setError('Network error. Please check if the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleOAuthLogin = (provider) => {
    // For now, show a message. In production, this would redirect to OAuth flow
    setError(`OAuth login with ${provider} will be implemented in the next phase`);
  };

  return (
    <div className="auth-container-blue">
      {/* Floating particles/elements */}
      <div className="floating-elements">
        <div className="floating-element" style={{top: '10%', left: '10%', animationDelay: '0s'}}>🎯</div>
        <div className="floating-element" style={{top: '20%', right: '15%', animationDelay: '2s'}}>🤖</div>
        <div className="floating-element" style={{bottom: '20%', left: '20%', animationDelay: '4s'}}>👁️</div>
        <div className="floating-element" style={{bottom: '10%', right: '10%', animationDelay: '6s'}}>🏆</div>
      </div>

      <div className="auth-card-blue">
        <div className="card-header">
          <div className="logo-section">
            <div className="logo-badge">
              <span className="logo-text">EyeVsAI</span>
              <div className="logo-animation">
                <div className="pulse-ring"></div>
                <div className="pulse-ring" style={{animationDelay: '1s'}}></div>
              </div>
            </div>
          </div>
          
          <h1 className="main-title">
            {isLogin ? 'Welcome Back, Player!' : 'Join the Battle!'}
          </h1>
          <p className="subtitle">
            {isLogin ? 'Ready to challenge the machines?' : 'Start your AI competition journey'}
          </p>
        </div>

        <div className="auth-mode-switch">
          <div className="switch-container">
            <input
              type="radio"
              id="login-mode"
              name="auth-mode"
              checked={isLogin}
              onChange={() => setIsLogin(true)}
            />
            <label htmlFor="login-mode" className="switch-label">
              <span className="switch-icon">🔑</span>
              Sign In
            </label>
            
            <input
              type="radio"
              id="signup-mode"
              name="auth-mode"
              checked={!isLogin}
              onChange={() => setIsLogin(false)}
            />
            <label htmlFor="signup-mode" className="switch-label">
              <span className="switch-icon">⚡</span>
              Sign Up
            </label>
            
            <div className="switch-indicator"></div>
          </div>
        </div>

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form-blue">
          {!isLogin && (
            <div className="form-field">
              <div className="field-icon">👤</div>
              <input
                type="text"
                name="displayName"
                placeholder="Display Name"
                value={formData.displayName}
                onChange={handleInputChange}
                className="field-input"
                disabled={loading}
              />
            </div>
          )}
          
          <div className="form-field">
            <div className="field-icon">📧</div>
            <input
              type="email"
              name="email"
              placeholder="Email Address"
              value={formData.email}
              onChange={handleInputChange}
              className="field-input"
              required
              disabled={loading}
            />
          </div>

          <div className="form-field">
            <div className="field-icon">🔒</div>
            <input
              type="password"
              name="password"
              placeholder="Password"
              value={formData.password}
              onChange={handleInputChange}
              className="field-input"
              required
              disabled={loading}
            />
          </div>

          <button type="submit" className="submit-button" disabled={loading}>
            <span className="button-text">
              {loading ? 'Processing...' : (isLogin ? 'Enter the Arena' : 'Start Your Journey')}
            </span>
            <div className="button-animation">
              <div className="button-shine"></div>
            </div>
          </button>
        </form>

        <div className="divider-section">
          <div className="divider-line"></div>
          <span className="divider-text">or connect with</span>
          <div className="divider-line"></div>
        </div>

        <div className="social-login-grid">
          <button 
            className="social-btn-card google-card"
            onClick={() => handleOAuthLogin('Google')}
            disabled={loading}
          >
            <div className="social-icon-container">
              <svg className="social-svg" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
            </div>
            <span>Google</span>
          </button>

          <button 
            className="social-btn-card facebook-card"
            onClick={() => handleOAuthLogin('Facebook')}
            disabled={loading}
          >
            <div className="social-icon-container">
              <svg className="social-svg" viewBox="0 0 24 24">
                <path fill="#1877F2" d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
              </svg>
            </div>
            <span>Facebook</span>
          </button>

          <button 
            className="social-btn-card github-card"
            onClick={() => handleOAuthLogin('GitHub')}
            disabled={loading}
          >
            <div className="social-icon-container">
              <svg className="social-svg" viewBox="0 0 24 24">
                <path fill="currentColor" d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
            </div>
            <span>GitHub</span>
          </button>

          <button 
            className="social-btn-card discord-card"
            onClick={() => handleOAuthLogin('Discord')}
            disabled={loading}
          >
            <div className="social-icon-container">
              <svg className="social-svg" viewBox="0 0 24 24">
                <path fill="#5865F2" d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
              </svg>
            </div>
            <span>Discord</span>
          </button>
        </div>

        <div className="guest-access">
          <button 
            className="guest-btn-blue"
            onClick={handleGuestLogin}
            disabled={loading}
          >
            <div className="guest-icon">👻</div>
            <div className="guest-content">
              <span className="guest-title">Quick Play</span>
              <span className="guest-desc">Try without signing up</span>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;