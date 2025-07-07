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

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API functions
export const apiService = {
  // Health check
  healthCheck: () => api.get('/health'),
  
  // Model management
  getModelsStatus: () => api.get('/models/status'),
  getModels: () => api.get('/models'),
  
  // Predictions
  predictImage: (imageData, modelType = 'ensemble', topK = 3) =>
    api.post('/predict', {
      image_data: imageData,
      model_type: modelType,
      top_k: topK,
    }),
  
  predictUploadedImage: (file, modelType = 'ensemble', topK = 3) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);
    formData.append('top_k', topK);
    
    return api.post('/predict/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Game functions
  createGameChallenge: (category, difficulty) =>
    api.post('/game/challenge', {
      category,
      difficulty,
    }),
  
  submitGameAnswer: (challengeId, userAnswer) =>
    api.post('/game/submit', {
      challenge_id: challengeId,
      user_answer: userAnswer,
    }),
};

export default api;