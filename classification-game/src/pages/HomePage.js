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

import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Box,
  Chip,
  CircularProgress,
} from '@mui/material';
import { Link } from 'react-router-dom';
import { apiService } from '../services/api';

const HomePage = () => {
  const [modelsStatus, setModelsStatus] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModelsStatus = async () => {
      try {
        const response = await apiService.getModelsStatus();
        setModelsStatus(response.data);
      } catch (error) {
        console.error('Failed to fetch models status:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchModelsStatus();
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'loaded':
        return 'success';
      case 'loading':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box textAlign="center" mb={4}>
        <Typography variant="h2" component="h1" gutterBottom>
          Welcome to the AI Image Classification Game
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          Test your image recognition skills against multiple AI models
        </Typography>
      </Box>

      {/* Models Status */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            AI Models Status
          </Typography>
          {loading ? (
            <Box display="flex" justifyContent="center">
              <CircularProgress />
            </Box>
          ) : (
            <Box display="flex" gap={1} flexWrap="wrap">
              {Object.entries(modelsStatus).map(([model, status]) => (
                <Chip
                  key={model}
                  label={`${model}: ${status}`}
                  color={getStatusColor(status)}
                  variant="outlined"
                />
              ))}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Game Options */}
      <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h5" component="h2" gutterBottom>
                Challenge Mode
              </Typography>
              <Typography color="text.secondary" paragraph>
                Compete against AI models in identifying dog breeds, cat breeds, 
                fruits, and vegetables. See if you can beat the machine learning algorithms!
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                size="large"
                variant="contained"
                component={Link}
                to="/game"
                fullWidth
              >
                Start Challenge
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h5" component="h2" gutterBottom>
                Upload & Test
              </Typography>
              <Typography color="text.secondary" paragraph>
                Upload your own images and see how different AI models classify them. 
                Compare predictions from shallow learning, deep learning, and transfer learning models.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                size="large"
                variant="contained"
                component={Link}
                to="/upload"
                fullWidth
              >
                Upload Image
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h5" component="h2" gutterBottom>
                Leaderboard
              </Typography>
              <Typography color="text.secondary" paragraph>
                Check out the top performers and see how you rank against other players. 
                Track your progress and compare your accuracy with AI models.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                size="large"
                variant="contained"
                component={Link}
                to="/leaderboard"
                fullWidth
              >
                View Rankings
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>

      {/* About Section */}
      <Box mt={6}>
        <Typography variant="h4" component="h2" gutterBottom>
          About the Models
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  Shallow Learning
                </Typography>
                <Typography variant="body2">
                  Traditional machine learning approaches using feature extraction 
                  and classical algorithms.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  Deep Learning v1
                </Typography>
                <Typography variant="body2">
                  First neural network implementation with custom architecture 
                  trained from scratch.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  Deep Learning v2
                </Typography>
                <Typography variant="body2">
                  Improved neural network with enhanced architecture and 
                  optimization techniques.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  Transfer Learning
                </Typography>
                <Typography variant="body2">
                  Pre-trained model fine-tuned with additional layers for 
                  specialized classification tasks.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default HomePage;