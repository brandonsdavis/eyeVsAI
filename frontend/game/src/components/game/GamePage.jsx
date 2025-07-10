import { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import './GamePage.css';

const GamePage = () => {
  const { user, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  
  // Game state
  const [gameState, setGameState] = useState('setup'); // 'setup', 'playing', 'round-result', 'game-complete'
  const [gameData, setGameData] = useState({
    datasets: {},
    selectedDataset: '',
    selectedDifficulty: '',
    selectedModel: '',
    session: null,
    currentRound: null,
    roundResults: [],
    gameResults: null
  });
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedAnswer, setSelectedAnswer] = useState('');
  const [timeLeft, setTimeLeft] = useState(30);
  const [showAIReveal, setShowAIReveal] = useState(false);

  // Redirect if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/auth');
    }
  }, [isAuthenticated, navigate]);

  // Fetch available datasets on component mount
  useEffect(() => {
    if (isAuthenticated) {
      fetchDatasets();
    }
  }, [isAuthenticated]);

  // Timer for rounds
  useEffect(() => {
    let timer;
    if (gameState === 'playing' && timeLeft > 0) {
      timer = setTimeout(() => {
        setTimeLeft(timeLeft - 1);
      }, 1000);
    } else if (gameState === 'playing' && timeLeft === 0) {
      // Time's up - auto submit
      handleAnswerSubmit();
    }
    return () => clearTimeout(timer);
  }, [gameState, timeLeft]);

  const fetchDatasets = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/game/datasets');
      if (response.ok) {
        const data = await response.json();
        setGameData(prev => ({ ...prev, datasets: data.datasets }));
      }
    } catch (err) {
      setError('Failed to load game datasets');
    }
  };

  const startGameSession = async () => {
    if (!gameData.selectedDataset || !gameData.selectedDifficulty || !gameData.selectedModel) {
      setError('Please select dataset, difficulty, and AI model');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch('http://localhost:8000/api/v1/game/session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          dataset: gameData.selectedDataset,
          difficulty: gameData.selectedDifficulty,
          ai_model_key: gameData.selectedModel,
          total_rounds: 10
        })
      });

      if (response.ok) {
        const session = await response.json();
        setGameData(prev => ({ ...prev, session }));
        startNextRound(session.session_id);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to start game session');
      }
    } catch (err) {
      setError('Network error. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  const startNextRound = async (sessionId) => {
    setLoading(true);
    setSelectedAnswer('');
    setTimeLeft(30);
    setShowAIReveal(false);

    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`http://localhost:8000/api/v1/game/session/${sessionId}/round`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const roundData = await response.json();
        setGameData(prev => ({ ...prev, currentRound: roundData }));
        setGameState('playing');
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to start round');
      }
    } catch (err) {
      setError('Network error. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerSubmit = async () => {
    if (!selectedAnswer && timeLeft > 0) {
      setError('Please select an answer');
      return;
    }

    setLoading(true);
    const startTime = Date.now();

    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`http://localhost:8000/api/v1/game/round/${gameData.currentRound.round_id}/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          round_id: gameData.currentRound.round_id,
          user_answer: selectedAnswer || 'timeout',
          response_time_ms: 30000 - (timeLeft * 1000)
        })
      });

      if (response.ok) {
        const result = await response.json();
        setGameData(prev => ({ 
          ...prev, 
          roundResults: [...prev.roundResults, result]
        }));
        setGameState('round-result');
        setShowAIReveal(true);
        
        // Auto-advance to next round after 5 seconds
        setTimeout(() => {
          if (gameData.session.current_round >= gameData.session.total_rounds) {
            completeGame();
          } else {
            startNextRound(gameData.session.session_id);
          }
        }, 5000);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to submit answer');
      }
    } catch (err) {
      setError('Network error. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  const completeGame = async () => {
    setLoading(true);

    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`http://localhost:8000/api/v1/game/session/${gameData.session.session_id}/complete`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const results = await response.json();
        setGameData(prev => ({ ...prev, gameResults: results }));
        setGameState('game-complete');
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to complete game');
      }
    } catch (err) {
      setError('Network error. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  const resetGame = () => {
    setGameState('setup');
    setGameData(prev => ({
      ...prev,
      session: null,
      currentRound: null,
      roundResults: [],
      gameResults: null
    }));
    setSelectedAnswer('');
    setTimeLeft(30);
    setShowAIReveal(false);
    setError('');
  };

  if (!isAuthenticated) {
    return <div>Loading...</div>;
  }

  return (
    <div className="game-container">
      {/* Floating background elements */}
      <div className="game-bg-elements">
        <div className="bg-element" style={{top: '10%', left: '5%', animationDelay: '0s'}}>🎯</div>
        <div className="bg-element" style={{top: '20%', right: '10%', animationDelay: '2s'}}>🧠</div>
        <div className="bg-element" style={{bottom: '15%', left: '15%', animationDelay: '4s'}}>⚡</div>
        <div className="bg-element" style={{bottom: '25%', right: '5%', animationDelay: '6s'}}>🏆</div>
      </div>

      <div className="game-content">
        {/* Header */}
        <div className="game-header">
          <div className="game-title">
            <h1>EyeVsAI Challenge</h1>
            <p>Welcome, {user?.display_name || 'Player'}!</p>
          </div>
          
          {gameState !== 'setup' && (
            <div className="game-progress">
              <div className="progress-info">
                <span>Round {gameData.currentRound?.round_number || 0} / {gameData.session?.total_rounds || 10}</span>
                <span>Score: {gameData.session?.total_score || 0}</span>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ width: `${((gameData.currentRound?.round_number || 0) / (gameData.session?.total_rounds || 10)) * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            {error}
            <button onClick={() => setError('')} className="error-close">×</button>
          </div>
        )}

        {/* Game Setup */}
        {gameState === 'setup' && (
          <div className="game-setup">
            <div className="setup-card">
              <h2>Choose Your Challenge</h2>
              
              <div className="setup-section">
                <h3>Dataset</h3>
                <div className="dataset-grid">
                  {Object.entries(gameData.datasets).map(([key, dataset]) => (
                    <div 
                      key={key}
                      className={`dataset-card ${gameData.selectedDataset === key ? 'selected' : ''}`}
                      onClick={() => setGameData(prev => ({ ...prev, selectedDataset: key, selectedDifficulty: '', selectedModel: '' }))}
                    >
                      <h4>{dataset.name}</h4>
                      <p>{dataset.description}</p>
                      <div className="dataset-stats">
                        <span>{dataset.num_classes} classes</span>
                        <span>{dataset.total_models_available} AI models</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {gameData.selectedDataset && (
                <div className="setup-section">
                  <h3>Difficulty Level</h3>
                  <div className="difficulty-grid">
                    {Object.entries(gameData.datasets[gameData.selectedDataset].difficulty_levels).map(([level, data]) => (
                      <div 
                        key={level}
                        className={`difficulty-card ${gameData.selectedDifficulty === level ? 'selected' : ''} ${level}`}
                        onClick={() => setGameData(prev => ({ ...prev, selectedDifficulty: level, selectedModel: '' }))}
                      >
                        <h4>{level.toUpperCase()}</h4>
                        <p>{data.description}</p>
                        <div className="difficulty-stats">
                          <span>{data.models.length} AI opponents</span>
                          <span>{Math.round(data.accuracy_range[0] * 100)}% - {Math.round(data.accuracy_range[1] * 100)}% accuracy</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {gameData.selectedDataset && gameData.selectedDifficulty && (
                <div className="setup-section">
                  <h3>Choose Your AI Opponent</h3>
                  <div className="model-grid">
                    {gameData.datasets[gameData.selectedDataset].difficulty_levels[gameData.selectedDifficulty].models.map((model, idx) => (
                      <div 
                        key={idx}
                        className={`model-card ${gameData.selectedModel === model.model_key ? 'selected' : ''}`}
                        onClick={() => setGameData(prev => ({ ...prev, selectedModel: model.model_key }))}
                      >
                        <h4>{model.name}</h4>
                        <div className="model-stats">
                          <span className="accuracy">{Math.round(model.accuracy * 100)}% accuracy</span>
                          <span className="version">{model.version}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {gameData.selectedDataset && gameData.selectedDifficulty && gameData.selectedModel && (
                <div className="setup-actions">
                  <button 
                    className="start-game-btn"
                    onClick={startGameSession}
                    disabled={loading}
                  >
                    {loading ? 'Starting Battle...' : 'Start Challenge'}
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Game Playing */}
        {gameState === 'playing' && gameData.currentRound && (
          <div className="game-playing">
            <div className="game-round">
              <div className="round-header">
                <h2>Round {gameData.currentRound.round_number}</h2>
                <div className="timer">
                  <span className="timer-text">Time: {timeLeft}s</span>
                  <div className="timer-bar">
                    <div 
                      className="timer-fill"
                      style={{ width: `${(timeLeft / 30) * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              <div className="image-container">
                <img 
                  src={`http://localhost:8000${gameData.currentRound.image_url}`}
                  alt="Classify this image"
                  className="game-image"
                />
              </div>

              <div className="answer-options">
                <h3>What do you see?</h3>
                <div className="options-grid">
                  {gameData.currentRound.options.map((option, idx) => (
                    <button
                      key={idx}
                      className={`option-btn ${selectedAnswer === option ? 'selected' : ''}`}
                      onClick={() => setSelectedAnswer(option)}
                      disabled={loading}
                    >
                      {option.replace(/_/g, ' ')}
                    </button>
                  ))}
                </div>
              </div>

              <div className="round-actions">
                <button 
                  className="submit-answer-btn"
                  onClick={handleAnswerSubmit}
                  disabled={loading || !selectedAnswer}
                >
                  {loading ? 'Submitting...' : 'Submit Answer'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Round Result */}
        {gameState === 'round-result' && gameData.roundResults.length > 0 && (
          <div className="round-result">
            <div className="result-card">
              {(() => {
                const lastResult = gameData.roundResults[gameData.roundResults.length - 1];
                return (
                  <>
                    <div className={`result-header ${lastResult.user_correct ? 'correct' : 'incorrect'}`}>
                      <h2>{lastResult.user_correct ? '🎉 Correct!' : '❌ Incorrect'}</h2>
                      <div className="points">+{lastResult.points_earned} points</div>
                    </div>

                    <div className="result-details">
                      <div className="answer-comparison">
                        <div className="your-answer">
                          <h4>Your Answer</h4>
                          <div className={`answer-value ${lastResult.user_correct ? 'correct' : 'incorrect'}`}>
                            {lastResult.user_answer.replace(/_/g, ' ')}
                          </div>
                        </div>
                        
                        <div className="vs-divider">VS</div>
                        
                        <div className="ai-answer">
                          <h4>AI's Answer</h4>
                          <div className={`answer-value ${lastResult.ai_answer === lastResult.correct_answer ? 'correct' : 'incorrect'}`}>
                            {lastResult.ai_answer.replace(/_/g, ' ')}
                          </div>
                          <div className="confidence">
                            {Math.round(lastResult.ai_confidence * 100)}% confident
                          </div>
                        </div>
                      </div>

                      <div className="correct-answer">
                        <h4>Correct Answer</h4>
                        <div className="answer-value correct">
                          {lastResult.correct_answer.replace(/_/g, ' ')}
                        </div>
                      </div>

                      <div className="explanation">
                        <p>{lastResult.explanation}</p>
                      </div>
                    </div>

                    <div className="result-actions">
                      <p>Next round starting automatically...</p>
                    </div>
                  </>
                );
              })()}
            </div>
          </div>
        )}

        {/* Game Complete */}
        {gameState === 'game-complete' && gameData.gameResults && (
          <div className="game-complete">
            <div className="results-card">
              <div className="results-header">
                <h2>🏆 Challenge Complete!</h2>
                <div className="final-score">{gameData.gameResults.final_score} points</div>
              </div>

              <div className="results-stats">
                <div className="stat-card">
                  <h4>Your Performance</h4>
                  <div className="stat-value">{gameData.gameResults.correct_answers}/{gameData.gameResults.total_rounds}</div>
                  <div className="stat-label">{Math.round(gameData.gameResults.accuracy * 100)}% accuracy</div>
                </div>
                
                <div className="stat-card">
                  <h4>AI Performance</h4>
                  <div className="stat-value">{gameData.gameResults.ai_correct_answers}/{gameData.gameResults.total_rounds}</div>
                  <div className="stat-label">{Math.round(gameData.gameResults.ai_accuracy * 100)}% accuracy</div>
                </div>
                
                <div className="stat-card">
                  <h4>Result</h4>
                  <div className={`stat-value ${gameData.gameResults.beat_ai ? 'victory' : 'defeat'}`}>
                    {gameData.gameResults.beat_ai ? '🎉 Victory!' : '🤖 AI Wins'}
                  </div>
                  <div className="stat-label">
                    {gameData.gameResults.beat_ai ? 'You beat the AI!' : 'Keep practicing!'}
                  </div>
                </div>
              </div>

              <div className="results-actions">
                <button className="play-again-btn" onClick={resetGame}>
                  Play Again
                </button>
                <button className="leaderboard-btn" onClick={() => navigate('/leaderboard')}>
                  View Leaderboard
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GamePage;