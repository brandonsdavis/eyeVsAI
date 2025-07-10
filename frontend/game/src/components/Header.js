/*
 * Copyright 2025 Brandon Davis
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React, { useState } from 'react';
import { AppBar, Toolbar, Typography, Button, Box, Menu, MenuItem, Avatar, IconButton, Chip } from '@mui/material';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const Header = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout, isAuthenticated } = useAuth();
  const [anchorEl, setAnchorEl] = useState(null);

  const isActive = (path) => location.pathname === path;

  const handleMenuClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    handleMenuClose();
    navigate('/');
  };

  const handleProfile = () => {
    navigate('/profile');
    handleMenuClose();
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          AI Image Classification Game
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Button
            color="inherit"
            component={Link}
            to="/"
            sx={{ 
              textDecoration: 'none',
              backgroundColor: isActive('/') ? 'rgba(255,255,255,0.1)' : 'transparent'
            }}
          >
            Home
          </Button>
          <Button
            color="inherit"
            component={Link}
            to="/game"
            sx={{ 
              textDecoration: 'none',
              backgroundColor: isActive('/game') ? 'rgba(255,255,255,0.1)' : 'transparent'
            }}
          >
            Play Game
          </Button>
          <Button
            color="inherit"
            component={Link}
            to="/upload"
            sx={{ 
              textDecoration: 'none',
              backgroundColor: isActive('/upload') ? 'rgba(255,255,255,0.1)' : 'transparent'
            }}
          >
            Upload & Test
          </Button>
          <Button
            color="inherit"
            component={Link}
            to="/leaderboard"
            sx={{ 
              textDecoration: 'none',
              backgroundColor: isActive('/leaderboard') ? 'rgba(255,255,255,0.1)' : 'transparent'
            }}
          >
            Leaderboard
          </Button>
          
          {isAuthenticated ? (
            <>
              <Chip
                avatar={<Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)' }}>{user?.display_name?.[0] || 'U'}</Avatar>}
                label={user?.display_name || 'User'}
                variant="outlined"
                sx={{ 
                  color: 'white', 
                  borderColor: 'rgba(255,255,255,0.3)',
                  '&:hover': { borderColor: 'rgba(255,255,255,0.5)' }
                }}
                onClick={handleMenuClick}
              />
              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
              >
                <MenuItem onClick={handleProfile}>Profile Settings</MenuItem>
                <MenuItem onClick={handleLogout}>Logout</MenuItem>
              </Menu>
            </>
          ) : (
            <Button
              color="inherit"
              component={Link}
              to="/auth"
              sx={{ 
                textDecoration: 'none',
                backgroundColor: isActive('/auth') ? 'rgba(255,255,255,0.1)' : 'transparent'
              }}
            >
              Login
            </Button>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;