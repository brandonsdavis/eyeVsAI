# Eye vs AI Frontend

A React-based web application that challenges users to compete against various AI models in image classification tasks. The game presents a fun, interactive way to explore how different machine learning models perform on real-world image recognition challenges.

## Overview

The frontend provides an engaging game experience where players:
- Select from multiple difficulty levels and datasets
- Compete against AI models in timed classification challenges
- Track their performance with detailed statistics
- Learn about different ML architectures through gameplay

## Architecture

### Technology Stack
- **React 18**: Modern React with hooks and functional components
- **React Router**: Client-side routing for navigation
- **Axios**: HTTP client for API communication
- **CSS Modules**: Component-scoped styling
- **Docker**: Containerized deployment

### Project Structure

```
frontend/
├── game/
│   ├── public/           # Static assets
│   ├── src/
│   │   ├── components/   # React components
│   │   │   ├── game/     # Game-specific components
│   │   │   ├── auth/     # Authentication components
│   │   │   └── common/   # Shared components
│   │   ├── services/     # API service layer
│   │   ├── utils/        # Helper functions
│   │   ├── App.js        # Main application component
│   │   └── index.js      # Application entry point
│   ├── package.json      # Dependencies and scripts
│   └── Dockerfile        # Container configuration
└── README.md            # This file
```

## Game Flow

### 1. Main Menu
Players start at the main menu where they can:
- View game instructions
- Select play mode (Guest or Authenticated)
- Access leaderboards
- Learn about the AI models

### 2. Game Configuration
Before starting, players choose:
- **Dataset**: What type of images to classify (pets, vegetables, instruments, etc.)
- **Difficulty**: Easy, Medium, or Hard (determines AI opponent strength)
- **Game Mode**: Standard (10 rounds) or Quick Play (5 rounds)

### 3. Gameplay
During each round:
1. An image is displayed to both player and AI
2. Multiple choice options appear (4 possible classifications)
3. Player selects their answer
4. AI makes its prediction (with simulated "thinking" time)
5. Results are revealed with correct answer highlighted
6. Points awarded based on accuracy and speed

### 4. Results
After completing all rounds:
- Final scores displayed
- Performance statistics shown
- Option to play again or return to menu
- Scores saved to leaderboard (if authenticated)

## Component Architecture

### Core Components

#### GameBoard
Main game container managing game state and flow.
```javascript
// Handles game logic, scoring, and round progression
const GameBoard = ({ config, onGameEnd }) => {
  // Game state management
  // Round progression logic
  // Score calculation
}
```

#### ImageDisplay
Displays the current image to classify with loading states.
```javascript
// Responsive image display with loading indicators
const ImageDisplay = ({ imageUrl, isLoading }) => {
  // Image loading and error handling
  // Responsive sizing
}
```

#### AnswerPanel
Presents multiple choice options with selection handling.
```javascript
// Interactive answer selection with feedback
const AnswerPanel = ({ options, onSelect, disabled, correctAnswer }) => {
  // Option rendering
  // Selection feedback
  // Correct/incorrect indicators
}
```

#### ScoreBoard
Real-time score tracking and display.
```javascript
// Live score updates and statistics
const ScoreBoard = ({ playerScore, aiScore, round, totalRounds }) => {
  // Score visualization
  // Progress tracking
}
```

## API Integration

The frontend communicates with the backend API for all game functionality:

### Service Layer
Located in `src/services/api.js`:

```javascript
// Base configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Game endpoints
export const gameAPI = {
  // Start new game session
  startGame: (config) => axios.post('/api/game/start', config),
  
  // Get next round data
  getNextRound: (gameId) => axios.get(`/api/game/${gameId}/next`),
  
  // Submit answer
  submitAnswer: (gameId, answer) => axios.post(`/api/game/${gameId}/answer`, answer),
  
  // Get game results
  getResults: (gameId) => axios.get(`/api/game/${gameId}/results`)
};

// Model information
export const modelAPI = {
  // Get available models by difficulty
  getModels: (difficulty) => axios.get(`/api/models?difficulty=${difficulty}`),
  
  // Get model details
  getModelInfo: (modelId) => axios.get(`/api/models/${modelId}`)
};
```

### Error Handling
Comprehensive error handling for network issues:
- Automatic retry for failed requests
- User-friendly error messages
- Fallback UI states
- Offline mode detection

## State Management

The application uses React's built-in state management with Context API for global state:

### Game Context
Manages overall game state across components:
```javascript
const GameContext = createContext({
  gameConfig: null,
  currentGame: null,
  player: null,
  // ... other global state
});
```

### Local Component State
Individual components manage their own state using hooks:
- `useState`: Component-specific state
- `useEffect`: Side effects and API calls
- `useCallback`: Optimized event handlers
- `useMemo`: Expensive computations

## Styling Approach

The project uses CSS Modules for component isolation:

```css
/* GameBoard.module.css */
.container {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  padding: 2rem;
}

.gameArea {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

@media (max-width: 768px) {
  .gameArea {
    grid-template-columns: 1fr;
  }
}
```

### Design Principles
- Mobile-first responsive design
- Consistent color palette and typography
- Smooth animations and transitions
- Accessible UI with ARIA labels
- Dark mode support (future enhancement)

## Build and Deployment

### Development
```bash
# Install dependencies
npm install

# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build
```

### Environment Variables
Configure in `.env` or `.env.local`:
```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_GAME_TITLE=Eye vs AI
REACT_APP_VERSION=1.0.0
```

### Docker Deployment
```bash
# Build Docker image
docker build -t eyevsai-frontend .

# Run container
docker run -p 3000:80 eyevsai-frontend
```

The Dockerfile uses multi-stage build for optimization:
1. Build stage: Compile React application
2. Production stage: Serve with nginx

## Performance Optimization

### Code Splitting
Dynamic imports for route-based splitting:
```javascript
const GameView = lazy(() => import('./views/GameView'));
const LeaderboardView = lazy(() => import('./views/LeaderboardView'));
```

### Image Optimization
- Lazy loading for images
- Progressive loading with blur placeholders
- Responsive image sizing
- WebP format support

### Bundle Optimization
- Tree shaking for unused code removal
- Minification and compression
- Asset caching strategies
- CDN integration for static assets

## Testing Strategy

### Unit Tests
Component testing with React Testing Library:
```javascript
describe('AnswerPanel', () => {
  it('should disable options after selection', () => {
    // Test implementation
  });
});
```

### Integration Tests
API integration and user flow testing:
```javascript
describe('Game Flow', () => {
  it('should complete a full game session', async () => {
    // Test implementation
  });
});
```

### E2E Tests (Future)
Cypress tests for critical user paths:
- Complete game playthrough
- Authentication flow
- Leaderboard interaction

## Future Enhancements

### Planned Features
1. **Multiplayer Mode**: Real-time competition between players
2. **Tutorial Mode**: Interactive learning about ML concepts
3. **Custom Datasets**: User-uploaded images for classification

### Production Considerations
For a production deployment at scale:
1. **CDN Integration**: CloudFront for global asset delivery
2. **Error Tracking**: Sentry for production error monitoring
3. **Performance Monitoring**: Web Vitals tracking
4. **A/B Testing**: Feature flag system for gradual rollouts
5. **Load Testing**: Ensure frontend handles concurrent users

## Development Guidelines

### Code Style
- Functional components with hooks
- PropTypes for type checking
- ESLint configuration for consistency
- Prettier for code formatting

### Component Guidelines
- Single responsibility principle
- Props documentation
- Error boundary implementation
- Accessibility compliance

### Git Workflow
- Feature branches for new development
- Pull requests with code review
- Semantic commit messages
- Version tagging for releases

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check API_URL environment variable
   - Verify backend is running
   - Check CORS configuration

2. **Images Not Loading**
   - Verify image URLs from API
   - Check network tab for 404s
   - Ensure proper CORS headers

3. **Build Failures**
   - Clear node_modules and reinstall
   - Check Node version compatibility
   - Verify all dependencies resolved

4. **Performance Issues**
   - Check browser console for errors
   - Profile with React DevTools
   - Verify API response times

## Support

For issues or questions:
1. Check existing GitHub issues
2. Review API documentation
3. Consult backend README for API details
4. Contact development team