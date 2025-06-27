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

import React, { useState, useCallback } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Box,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { CloudUpload } from '@mui/icons-material';
import { apiService } from '../services/api';

const UploadPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [modelType, setModelType] = useState('ensemble');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [topK] = useState(5);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setPredictions(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      const response = await apiService.predictUploadedImage(
        selectedFile,
        modelType,
        topK
      );
      setPredictions(response.data);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setPredictions(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom textAlign="center">
        Upload & Test Images
      </Typography>
      <Typography variant="h6" color="text.secondary" textAlign="center" paragraph>
        Upload your own images and see how different AI models classify them
      </Typography>

      <Grid container spacing={4}>
        {/* Upload Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Upload Image
              </Typography>
              
              {/* Dropzone */}
              <Box
                {...getRootProps()}
                sx={{
                  border: '2px dashed #ccc',
                  borderRadius: 2,
                  p: 4,
                  textAlign: 'center',
                  cursor: 'pointer',
                  backgroundColor: isDragActive ? '#f5f5f5' : 'transparent',
                  '&:hover': {
                    backgroundColor: '#f9f9f9',
                  },
                }}
              >
                <input {...getInputProps()} />
                <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                {isDragActive ? (
                  <Typography>Drop the image here...</Typography>
                ) : (
                  <Typography>
                    Drag and drop an image here, or click to select
                  </Typography>
                )}
              </Box>

              {/* Preview */}
              {previewUrl && (
                <Box sx={{ mt: 3 }}>
                  <img
                    src={previewUrl}
                    alt="Preview"
                    style={{
                      width: '100%',
                      maxHeight: '300px',
                      objectFit: 'contain',
                      borderRadius: '8px',
                    }}
                  />
                  <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <Chip label={selectedFile?.name} variant="outlined" />
                    <Chip
                      label={`${(selectedFile?.size / 1024 / 1024).toFixed(2)} MB`}
                      variant="outlined"
                    />
                  </Box>
                </Box>
              )}

              {/* Model Selection */}
              <FormControl fullWidth sx={{ mt: 3 }}>
                <InputLabel>Model Type</InputLabel>
                <Select
                  value={modelType}
                  label="Model Type"
                  onChange={(e) => setModelType(e.target.value)}
                >
                  <MenuItem value="ensemble">Ensemble (All Models)</MenuItem>
                  <MenuItem value="shallow">Shallow Learning</MenuItem>
                  <MenuItem value="deep_v1">Deep Learning v1</MenuItem>
                  <MenuItem value="deep_v2">Deep Learning v2</MenuItem>
                  <MenuItem value="transfer">Transfer Learning</MenuItem>
                </Select>
              </FormControl>

              {/* Action Buttons */}
              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  onClick={handlePredict}
                  disabled={!selectedFile || loading}
                  fullWidth
                >
                  {loading ? 'Predicting...' : 'Predict'}
                </Button>
                <Button
                  variant="outlined"
                  onClick={clearImage}
                  disabled={!selectedFile}
                >
                  Clear
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Results Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Prediction Results
              </Typography>

              {loading && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography textAlign="center" sx={{ mt: 1 }}>
                    Processing image...
                  </Typography>
                </Box>
              )}

              {predictions && (
                <>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Model: {predictions.model_used}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Processing time: {predictions.processing_time.toFixed(3)}s
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Top confidence: {(predictions.confidence_score * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Typography variant="h6" gutterBottom>
                    Top Predictions:
                  </Typography>
                  
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Rank</TableCell>
                          <TableCell>Class</TableCell>
                          <TableCell>Confidence</TableCell>
                          <TableCell>Score</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {predictions.top_predictions.map(([className, confidence], index) => (
                          <TableRow key={className}>
                            <TableCell>{index + 1}</TableCell>
                            <TableCell>{className}</TableCell>
                            <TableCell>
                              <Chip
                                label={`${(confidence * 100).toFixed(1)}%`}
                                color={getConfidenceColor(confidence)}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <LinearProgress
                                  variant="determinate"
                                  value={confidence * 100}
                                  sx={{ width: 60, height: 4 }}
                                />
                                <Typography variant="caption">
                                  {(confidence * 100).toFixed(0)}%
                                </Typography>
                              </Box>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </>
              )}

              {!predictions && !loading && (
                <Box textAlign="center" sx={{ py: 4, color: 'text.secondary' }}>
                  <Typography>
                    Upload an image and click "Predict" to see results
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default UploadPage;