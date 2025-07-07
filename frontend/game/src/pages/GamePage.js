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
  Card,
  CardContent,
  CardMedia,
  Box,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
} from '@mui/material';
import Confetti from 'react-confetti';
import { apiService } from '../services/api';

const GamePage = () => {
  const [gameState, setGameState] = useState('setup'); // setup, playing, results
  const [category, setCategory] = useState('dogs_cats');
  const [difficulty, setDifficulty] = useState('easy');
  const [currentChallenge, setCurrentChallenge] = useState(null);
  const [selectedAnswer, setSelectedAnswer] = useState('');
  const [gameResult, setGameResult] = useState(null);
  const [score, setScore] = useState(0);
  const [round, setRound] = useState(1);
  const [totalRounds] = useState(5);
  const [loading, setLoading] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);

  const startGame = async () => {
    setLoading(true);
    setGameState('playing');
    setScore(0);
    setRound(1);
    await loadNextChallenge();
  };

  const loadNextChallenge = async () => {
    try {
      setLoading(true);
      const response = await apiService.createGameChallenge(category, difficulty);
      setCurrentChallenge(response.data);
      setSelectedAnswer('');
    } catch (error) {
      console.error('Failed to load challenge:', error);
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = async () => {
    if (!selectedAnswer || !currentChallenge) return;

    try {
      setLoading(true);
      const response = await apiService.submitGameAnswer(
        currentChallenge.challenge_id,
        selectedAnswer
      );
      
      setGameResult(response.data);
      
      if (response.data.correct) {
        setScore(score + response.data.user_score);
        setShowConfetti(true);
        setTimeout(() => setShowConfetti(false), 3000);
      }

      setTimeout(() => {
        if (round < totalRounds) {
          setRound(round + 1);
          setGameResult(null);
          loadNextChallenge();
        } else {
          setGameState('finished');
        }
      }, 3000);

    } catch (error) {
      console.error('Failed to submit answer:', error);
    } finally {
      setLoading(false);
    }
  };

  const resetGame = () => {
    setGameState('setup');
    setCurrentChallenge(null);
    setSelectedAnswer('');
    setGameResult(null);
    setScore(0);
    setRound(1);
  };

  const getScoreColor = () => {
    const percentage = (score / (totalRounds * 30)) * 100; // Max 30 points per hard question
    if (percentage >= 80) return 'success';
    if (percentage >= 60) return 'warning';
    return 'error';
  };

  if (gameState === 'setup') {
    return (
      <Container maxWidth="md" sx={{ mt: 4 }}>
        <Card>
          <CardContent>
            <Typography variant="h4" component="h1" gutterBottom textAlign="center">
              Challenge Setup
            </Typography>
            
            <Grid container spacing={3} sx={{ mt: 2 }}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Category</InputLabel>
                  <Select
                    value={category}
                    label="Category"
                    onChange={(e) => setCategory(e.target.value)}
                  >
                    <MenuItem value="dogs_cats">Dogs & Cats</MenuItem>
                    <MenuItem value="fruits_vegetables">Fruits & Vegetables</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Difficulty</InputLabel>
                  <Select
                    value={difficulty}
                    label="Difficulty"
                    onChange={(e) => setDifficulty(e.target.value)}
                  >
                    <MenuItem value="easy">Easy (10 pts)</MenuItem>
                    <MenuItem value="medium">Medium (20 pts)</MenuItem>
                    <MenuItem value="hard">Hard (30 pts)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>

            <Box textAlign="center" sx={{ mt: 4 }}>
              <Typography variant="body1" color="text.secondary" paragraph>
                You'll face {totalRounds} challenges. Try to beat the AI models!
              </Typography>
              <Button
                variant="contained"
                size="large"
                onClick={startGame}
                disabled={loading}
              >
                Start Game
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Container>
    );
  }

  if (gameState === 'finished') {
    return (
      <Container maxWidth="md" sx={{ mt: 4 }}>
        <Card>
          <CardContent textAlign="center">
            <Typography variant="h3" component="h1" gutterBottom>
              Game Complete!
            </Typography>
            <Typography variant="h4" color="primary" gutterBottom>
              Final Score: {score}
            </Typography>
            <Chip
              label={`${Math.round((score / (totalRounds * 30)) * 100)}% Accuracy`}
              color={getScoreColor()}
              size="large"
              sx={{ mt: 2, mb: 4 }}
            />
            <Box>
              <Button
                variant="contained"
                onClick={resetGame}
                sx={{ mr: 2 }}
              >
                Play Again
              </Button>
              <Button
                variant="outlined"
                onClick={() => window.location.href = '/leaderboard'}
              >
                View Leaderboard
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      {showConfetti && <Confetti />}
      
      {/* Progress */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Round {round} of {totalRounds} - Score: {score}
        </Typography>
        <LinearProgress
          variant="determinate"
          value={(round / totalRounds) * 100}
          sx={{ height: 8, borderRadius: 4 }}
        />
      </Box>

      {loading ? (
        <Card>
          <CardContent textAlign="center" sx={{ py: 8 }}>
            <LinearProgress sx={{ mb: 2 }} />
            <Typography>Loading challenge...</Typography>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent>
            {currentChallenge && (
              <>
                <CardMedia
                  component="img"
                  height="300"
                  image={currentChallenge.image_url}
                  alt="Challenge image"
                  sx={{ objectFit: 'contain', mb: 3 }}
                />
                
                <Typography variant="h6" gutterBottom>
                  What do you think this is?
                </Typography>
                
                <Grid container spacing={2}>
                  {currentChallenge.options.map((option, index) => (
                    <Grid item xs={12} sm={6} key={index}>
                      <Button
                        fullWidth
                        variant={selectedAnswer === option ? 'contained' : 'outlined'}
                        onClick={() => setSelectedAnswer(option)}
                        sx={{ py: 2 }}
                      >
                        {option}
                      </Button>
                    </Grid>
                  ))}
                </Grid>
                
                <Box textAlign="center" sx={{ mt: 3 }}>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={submitAnswer}
                    disabled={!selectedAnswer || loading}
                  >
                    Submit Answer
                  </Button>
                </Box>
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* Results Dialog */}
      <Dialog open={!!gameResult} maxWidth="sm" fullWidth>
        <DialogTitle>
          {gameResult?.correct ? 'Correct!' : 'Incorrect'}
        </DialogTitle>
        <DialogContent>
          <Typography paragraph>
            {gameResult?.explanation}
          </Typography>
          {gameResult?.correct && (
            <Typography color="primary" variant="h6">
              +{gameResult.user_score} points!
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setGameResult(null)} autoFocus>
            {round < totalRounds ? 'Next Round' : 'Finish Game'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default GamePage;