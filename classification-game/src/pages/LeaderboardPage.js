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
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Box,
  Chip,
  Grid,
  Avatar,
} from '@mui/material';
import { EmojiEvents, Person, SmartToy } from '@mui/icons-material';

const LeaderboardPage = () => {
  const [leaderboard, setLeaderboard] = useState([]);
  const [aiModelStats, setAiModelStats] = useState([]);

  useEffect(() => {
    // Mock data - in a real app this would come from an API
    const mockLeaderboard = [
      { rank: 1, name: 'Alex Chen', score: 2450, accuracy: 98, gamesPlayed: 25 },
      { rank: 2, name: 'Sarah Johnson', score: 2380, accuracy: 95, gamesPlayed: 24 },
      { rank: 3, name: 'Mike Rodriguez', score: 2250, accuracy: 92, gamesPlayed: 28 },
      { rank: 4, name: 'Emily Zhang', score: 2100, accuracy: 88, gamesPlayed: 22 },
      { rank: 5, name: 'David Kim', score: 2050, accuracy: 85, gamesPlayed: 26 },
      { rank: 6, name: 'Lisa Wang', score: 1980, accuracy: 83, gamesPlayed: 20 },
      { rank: 7, name: 'Tom Brown', score: 1920, accuracy: 80, gamesPlayed: 25 },
      { rank: 8, name: 'Anna Smith', score: 1850, accuracy: 78, gamesPlayed: 23 },
      { rank: 9, name: 'Chris Lee', score: 1800, accuracy: 75, gamesPlayed: 21 },
      { rank: 10, name: 'Maya Patel', score: 1750, accuracy: 72, gamesPlayed: 19 },
    ];

    const mockAiStats = [
      { name: 'Transfer Learning', accuracy: 94.2, avgConfidence: 0.89, category: 'All' },
      { name: 'Ensemble Model', accuracy: 92.8, avgConfidence: 0.91, category: 'All' },
      { name: 'Deep Learning v2', accuracy: 91.5, avgConfidence: 0.87, category: 'All' },
      { name: 'Deep Learning v1', accuracy: 88.3, avgConfidence: 0.84, category: 'All' },
      { name: 'Shallow Learning', accuracy: 82.7, avgConfidence: 0.78, category: 'All' },
    ];

    setLeaderboard(mockLeaderboard);
    setAiModelStats(mockAiStats);
  }, []);

  const getRankIcon = (rank) => {
    switch (rank) {
      case 1:
        return <EmojiEvents sx={{ color: '#FFD700' }} />;
      case 2:
        return <EmojiEvents sx={{ color: '#C0C0C0' }} />;
      case 3:
        return <EmojiEvents sx={{ color: '#CD7F32' }} />;
      default:
        return <Person />;
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 90) return 'success';
    if (accuracy >= 80) return 'warning';
    return 'error';
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom textAlign="center">
        Leaderboard
      </Typography>

      <Grid container spacing={4}>
        {/* Human Players Leaderboard */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Person sx={{ mr: 1 }} />
                <Typography variant="h5">Top Players</Typography>
              </Box>
              
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Rank</TableCell>
                      <TableCell>Player</TableCell>
                      <TableCell align="right">Score</TableCell>
                      <TableCell align="right">Accuracy</TableCell>
                      <TableCell align="right">Games</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {leaderboard.map((player) => (
                      <TableRow key={player.rank}>
                        <TableCell>
                          <Box display="flex" alignItems="center">
                            {getRankIcon(player.rank)}
                            <Typography sx={{ ml: 1 }}>{player.rank}</Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Box display="flex" alignItems="center">
                            <Avatar sx={{ mr: 2, width: 32, height: 32 }}>
                              {player.name.charAt(0)}
                            </Avatar>
                            <Typography>{player.name}</Typography>
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="h6" color="primary">
                            {player.score.toLocaleString()}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={`${player.accuracy}%`}
                            color={getAccuracyColor(player.accuracy)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="right">{player.gamesPlayed}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* AI Model Stats */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <SmartToy sx={{ mr: 1 }} />
                <Typography variant="h5">AI Model Performance</Typography>
              </Box>
              
              {aiModelStats.map((model, index) => (
                <Card key={model.name} variant="outlined" sx={{ mb: 2 }}>
                  <CardContent sx={{ pb: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      {model.name}
                    </Typography>
                    <Box display="flex" justifyContent="space-between" mb={1}>
                      <Typography variant="body2" color="text.secondary">
                        Accuracy:
                      </Typography>
                      <Chip
                        label={`${model.accuracy}%`}
                        color={getAccuracyColor(model.accuracy)}
                        size="small"
                      />
                    </Box>
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="body2" color="text.secondary">
                        Avg Confidence:
                      </Typography>
                      <Typography variant="body2">
                        {(model.avgConfidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              ))}
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Stats
              </Typography>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  Total Games Played:
                </Typography>
                <Typography variant="body2">1,247</Typography>
              </Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2" color="text.secondary">
                  Average Human Accuracy:
                </Typography>
                <Typography variant="body2">83.2%</Typography>
              </Box>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Best AI Model:
                </Typography>
                <Typography variant="body2">Transfer Learning</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default LeaderboardPage;