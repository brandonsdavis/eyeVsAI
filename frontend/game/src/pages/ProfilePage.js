import React, { useState } from 'react';
import { 
  Container, 
  Paper, 
  Typography, 
  TextField, 
  Button, 
  Box, 
  Alert,
  Divider,
  Chip,
  Grid
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

const ProfilePage = () => {
  const { user, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [displayName, setDisplayName] = useState(user?.display_name || '');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  // Redirect if not authenticated
  React.useEffect(() => {
    if (!isAuthenticated) {
      navigate('/auth');
    }
  }, [isAuthenticated, navigate]);

  const handleUpdateProfile = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage('');
    setError('');

    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch('http://localhost:8000/api/v1/auth/me', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          display_name: displayName
        })
      });

      if (response.ok) {
        setMessage('Profile updated successfully!');
        // Refresh the page to update the header
        window.location.reload();
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to update profile');
      }
    } catch (err) {
      setError('Network error. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  if (!isAuthenticated) {
    return <div>Loading...</div>;
  }

  return (
    <Container maxWidth="sm" sx={{ mt: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Profile Settings
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            Account Information
          </Typography>
          
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12}>
              <Typography variant="body2" color="text.secondary">
                Email
              </Typography>
              <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                {user?.email}
              </Typography>
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="body2" color="text.secondary">
                Account Type
              </Typography>
              <Chip 
                label={user?.is_guest ? 'Guest Account' : 'Registered Account'}
                color={user?.is_guest ? 'warning' : 'success'}
                size="small"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="body2" color="text.secondary">
                Member Since
              </Typography>
              <Typography variant="body1">
                {new Date(user?.created_at).toLocaleDateString()}
              </Typography>
            </Grid>
          </Grid>
        </Box>

        <Divider sx={{ my: 3 }} />

        <form onSubmit={handleUpdateProfile}>
          <Typography variant="h6" gutterBottom>
            Display Settings
          </Typography>
          
          <TextField
            fullWidth
            label="Display Name"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            margin="normal"
            required
            helperText="This name will be shown on leaderboards and in game"
            disabled={loading}
          />

          {message && (
            <Alert severity="success" sx={{ mt: 2 }}>
              {message}
            </Alert>
          )}

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}

          <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
            <Button
              type="submit"
              variant="contained"
              disabled={loading || !displayName.trim()}
            >
              {loading ? 'Updating...' : 'Update Profile'}
            </Button>
            
            <Button
              variant="outlined"
              onClick={() => navigate('/game')}
            >
              Back to Game
            </Button>
          </Box>
        </form>

        {user?.is_guest && (
          <Box sx={{ mt: 4 }}>
            <Divider sx={{ mb: 2 }} />
            <Alert severity="info">
              <Typography variant="body2">
                <strong>Guest Account:</strong> Your progress is saved locally but won't be preserved if you clear your browser data. 
                Consider creating a full account to secure your progress and leaderboard position.
              </Typography>
            </Alert>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default ProfilePage;