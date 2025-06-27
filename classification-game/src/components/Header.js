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

import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';

const Header = () => {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          AI Image Classification Game
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
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
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;