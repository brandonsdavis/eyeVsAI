# Copyright 2025 Brandon Davis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastapi import APIRouter, HTTPException, Depends, Request, status
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime, timedelta
import json
import os
import random
import hashlib
import secrets
import logging

from ..database import get_db
from ..auth import get_current_user, get_current_user_optional
from ..models import (
    GameConfiguration, GameSessionResponse, GameRoundResponse, 
    GameRoundSubmission, GameRoundResult, GameSessionSummary,
    LeaderboardResponse, LeaderboardEntry, GameDatasets
)
from ..db_models import User, GameSession, GameRound, Score
from ..services.game_backend_service import GameBackendService
from ..services.model_manager import ModelManager

router = APIRouter(prefix="/game", tags=["game"])
logger = logging.getLogger(__name__)


@router.get("/datasets", response_model=GameDatasets)
async def get_game_datasets():
    """Get available datasets and their configurations."""
    game_service = GameBackendService()
    datasets = await game_service.get_available_datasets()
    return GameDatasets(datasets=datasets)


@router.post("/session", response_model=GameSessionResponse)
async def create_game_session(
    config: GameConfiguration,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Create a new game session."""
    try:
        game_service = GameBackendService()
        
        # Validate configuration
        if not await game_service.validate_game_configuration(config):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid game configuration"
            )
        
        # Create game session
        session = GameSession(
            user_id=current_user.id if current_user else None,
            dataset=config.dataset,
            difficulty=config.difficulty,
            ai_model_key=config.ai_model_key,
            total_rounds=config.total_rounds,
            status="active"
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # Log session creation
        if current_user:
            from ..auth import auth_manager
            auth_manager.log_event(
                db=db,
                user_id=str(current_user.id),
                event_type="game_session_created",
                event_data={
                    "session_id": str(session.id),
                    "dataset": config.dataset,
                    "difficulty": config.difficulty,
                    "ai_model_key": config.ai_model_key
                },
                request=request
            )
        
        return GameSessionResponse(
            session_id=session.id,
            dataset=session.dataset,
            difficulty=session.difficulty,
            ai_model_key=session.ai_model_key,
            total_rounds=session.total_rounds,
            current_round=session.current_round,
            status=session.status,
            correct_answers=session.correct_answers,
            total_score=session.total_score,
            created_at=session.created_at
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create game session: {str(e)}"
        )


@router.get("/session/{session_id}", response_model=GameSessionResponse)
async def get_game_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get game session details."""
    session = db.query(GameSession).filter(GameSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game session not found"
        )
    
    # Check if user has access to this session
    if current_user and session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this game session"
        )
    
    return GameSessionResponse(
        session_id=session.id,
        dataset=session.dataset,
        difficulty=session.difficulty,
        ai_model_key=session.ai_model_key,
        total_rounds=session.total_rounds,
        current_round=session.current_round,
        status=session.status,
        correct_answers=session.correct_answers,
        total_score=session.total_score,
        created_at=session.created_at
    )


@router.post("/session/{session_id}/round", response_model=GameRoundResponse)
async def create_game_round(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Create a new game round."""
    try:
        session = db.query(GameSession).filter(GameSession.id == session_id).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game session not found"
            )
        
        if session.status != "active":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Game session is not active"
            )
        
        if session.current_round >= session.total_rounds:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All rounds completed"
            )
        
        # Check if user has access
        if current_user and session.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this game session"
            )
        
        game_service = GameBackendService()
        model_manager = ModelManager()
        
        # Get random image for this round
        image_data = await game_service.get_random_image(
            dataset=session.dataset,
            exclude_used=True,
            session_id=session_id
        )
        
        if not image_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No more images available for this dataset"
            )
        
        logger.info(f"Got image data: {image_data['image_path']}")
        
        # Get AI model and make prediction
        logger.info(f"Getting model: {session.ai_model_key}")
        model = await model_manager.get_model_by_key(session.ai_model_key)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI model not available"
            )
        
        logger.info(f"Got model: {model}")
        
        # AI makes prediction
        ai_prediction_result = await model.predict_image(image_data["image_path"])
        ai_prediction = ai_prediction_result["prediction"]
        ai_confidence = ai_prediction_result["confidence"]
        
        # Create cryptographic commitment for AI prediction
        commitment_salt = secrets.token_hex(32)
        commitment_data = f"{ai_prediction}:{ai_confidence}:{commitment_salt}"
        commitment_hash = hashlib.sha256(commitment_data.encode()).hexdigest()
        
        # Create round
        round_number = session.current_round + 1
        game_round = GameRound(
            session_id=session.id,
            round_number=round_number,
            image_path=image_data["image_path"],
            correct_answer=image_data["correct_answer"],
            ai_prediction=ai_prediction,
            ai_confidence=ai_confidence,
            ai_commitment_hash=commitment_hash,
            ai_commitment_salt=commitment_salt
        )
        
        db.add(game_round)
        
        # Update session
        session.current_round = round_number
        
        db.commit()
        db.refresh(game_round)
        
        # Get answer options
        options = await game_service.get_answer_options(
            dataset=session.dataset,
            correct_answer=image_data["correct_answer"]
        )
        
        return GameRoundResponse(
            round_id=str(game_round.id),
            round_number=round_number,
            image_url=f"/api/v1/images/{image_data['image_path']}",
            options=options,
            ai_commitment_hash=commitment_hash
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create game round: {str(e)}"
        )


@router.post("/round/{round_id}/submit", response_model=GameRoundResult)
async def submit_round_answer(
    round_id: str,
    submission: GameRoundSubmission,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Submit answer for a game round."""
    try:
        game_round = db.query(GameRound).filter(GameRound.id == round_id).first()
        
        if not game_round:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game round not found"
            )
        
        # Get session
        session = db.query(GameSession).filter(GameSession.id == game_round.session_id).first()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game session not found"
            )
        
        # Check if user has access
        if current_user and session.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this game round"
            )
        
        # Check if round already answered
        if game_round.user_answer is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Round already answered"
            )
        
        # Process answer
        user_correct = submission.user_answer == game_round.correct_answer
        
        # Calculate points (difficulty-based scoring)
        difficulty_multipliers = {"easy": 10, "medium": 20, "hard": 30}
        base_points = difficulty_multipliers.get(session.difficulty, 10)
        
        # Bonus for speed (if response time provided)
        speed_bonus = 0
        if submission.response_time_ms and submission.response_time_ms < 10000:  # Under 10 seconds
            speed_bonus = max(0, 5 - (submission.response_time_ms // 2000))  # 5 bonus points for < 2s, decreasing
        
        points_earned = (base_points + speed_bonus) if user_correct else 0
        
        # Update round
        game_round.user_answer = submission.user_answer
        game_round.user_correct = user_correct
        game_round.response_time_ms = submission.response_time_ms
        game_round.points_earned = points_earned
        game_round.answered_at = datetime.utcnow()
        
        # Update session stats
        if user_correct:
            session.correct_answers += 1
        session.total_score += points_earned
        
        db.commit()
        
        # Generate explanation
        explanation = generate_explanation(
            user_answer=submission.user_answer,
            correct_answer=game_round.correct_answer,
            ai_answer=game_round.ai_prediction,
            user_correct=user_correct,
            ai_confidence=game_round.ai_confidence
        )
        
        # Create commitment proof
        commitment_proof = f"{game_round.ai_prediction}:{game_round.ai_confidence}:{game_round.ai_commitment_salt}"
        
        return GameRoundResult(
            round_id=str(game_round.id),
            user_answer=submission.user_answer,
            correct_answer=game_round.correct_answer,
            user_correct=user_correct,
            ai_answer=game_round.ai_prediction,
            ai_confidence=game_round.ai_confidence,
            ai_commitment_proof=commitment_proof,
            points_earned=points_earned,
            explanation=explanation
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit round answer: {str(e)}"
        )


@router.post("/session/{session_id}/complete", response_model=GameSessionSummary)
async def complete_game_session(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Complete a game session and calculate final scores."""
    try:
        session = db.query(GameSession).filter(GameSession.id == session_id).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game session not found"
            )
        
        # Check if user has access
        if current_user and session.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this game session"
            )
        
        if session.status != "active":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Game session is not active"
            )
        
        # Get all rounds
        rounds = db.query(GameRound).filter(GameRound.session_id == session.id).all()
        
        # Calculate AI performance
        ai_correct_count = sum(1 for r in rounds if r.ai_prediction == r.correct_answer)
        ai_accuracy = ai_correct_count / len(rounds) if rounds else 0
        
        # Calculate user performance
        user_accuracy = session.correct_answers / session.total_rounds if session.total_rounds > 0 else 0
        beat_ai = user_accuracy > ai_accuracy
        
        # Calculate average response time
        response_times = [r.response_time_ms for r in rounds if r.response_time_ms]
        avg_response_time = sum(response_times) // len(response_times) if response_times else None
        
        # Update session
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        
        # Save score if user is registered
        if current_user and not current_user.is_guest:
            score = Score(
                user_id=current_user.id,
                session_id=session.id,
                dataset=session.dataset,
                difficulty=session.difficulty,
                ai_model_key=session.ai_model_key,
                total_score=session.total_score,
                correct_answers=session.correct_answers,
                total_rounds=session.total_rounds,
                accuracy=user_accuracy,
                average_response_time_ms=avg_response_time,
                ai_correct_answers=ai_correct_count,
                ai_accuracy=ai_accuracy,
                beat_ai=beat_ai
            )
            db.add(score)
        
        db.commit()
        
        # Log session completion
        if current_user:
            from ..auth import auth_manager
            auth_manager.log_event(
                db=db,
                user_id=str(current_user.id),
                event_type="game_session_completed",
                event_data={
                    "session_id": str(session.id),
                    "final_score": session.total_score,
                    "accuracy": user_accuracy,
                    "beat_ai": beat_ai
                },
                request=request
            )
        
        return GameSessionSummary(
            session_id=str(session.id),
            final_score=session.total_score,
            correct_answers=session.correct_answers,
            total_rounds=session.total_rounds,
            accuracy=user_accuracy,
            ai_correct_answers=ai_correct_count,
            ai_accuracy=ai_accuracy,
            beat_ai=beat_ai,
            average_response_time_ms=avg_response_time,
            completed_at=session.completed_at
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete game session: {str(e)}"
        )


@router.get("/leaderboard/{dataset}/{difficulty}", response_model=LeaderboardResponse)
async def get_leaderboard(
    dataset: str,
    difficulty: str,
    period: str = "all_time",  # "daily", "weekly", "monthly", "all_time"
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get leaderboard for a specific dataset and difficulty."""
    try:
        # Define time filters
        time_filters = {
            "daily": datetime.utcnow() - timedelta(days=1),
            "weekly": datetime.utcnow() - timedelta(days=7),
            "monthly": datetime.utcnow() - timedelta(days=30),
            "all_time": None
        }
        
        since_date = time_filters.get(period)
        if period not in time_filters:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid period. Must be 'daily', 'weekly', 'monthly', or 'all_time'"
            )
        
        # Base query
        query = db.query(
            Score.user_id,
            User.display_name,
            User.avatar_url,
            func.max(Score.total_score).label('best_score'),
            func.avg(Score.accuracy).label('avg_accuracy'),
            func.count(Score.id).label('games_played'),
            func.sum(func.cast(Score.beat_ai, db.bind.dialect.INTEGER)).label('beat_ai_count'),
            func.max(Score.created_at).label('latest_game')
        ).join(User).filter(
            Score.dataset == dataset,
            Score.difficulty == difficulty,
            User.is_guest == False
        )
        
        # Apply time filter
        if since_date:
            query = query.filter(Score.created_at >= since_date)
        
        # Group and order
        query = query.group_by(Score.user_id, User.display_name, User.avatar_url)
        query = query.order_by(desc('best_score'), desc('avg_accuracy'))
        query = query.limit(limit)
        
        results = query.all()
        
        # Build leaderboard entries
        entries = []
        for rank, result in enumerate(results, 1):
            entry = LeaderboardEntry(
                rank=rank,
                user_id=str(result.user_id),
                display_name=result.display_name,
                avatar_url=result.avatar_url,
                score=result.best_score,
                accuracy=float(result.avg_accuracy),
                games_played=result.games_played,
                beat_ai_count=result.beat_ai_count or 0,
                created_at=result.latest_game
            )
            entries.append(entry)
        
        # Find current user's rank
        user_rank = None
        if current_user and not current_user.is_guest:
            user_rank_query = db.query(func.count(Score.user_id).label('rank')).filter(
                Score.dataset == dataset,
                Score.difficulty == difficulty,
                Score.total_score > db.query(func.max(Score.total_score)).filter(
                    Score.user_id == current_user.id,
                    Score.dataset == dataset,
                    Score.difficulty == difficulty
                ).scalar_subquery()
            )
            
            if since_date:
                user_rank_query = user_rank_query.filter(Score.created_at >= since_date)
            
            rank_result = user_rank_query.scalar()
            user_rank = rank_result + 1 if rank_result is not None else None
        
        # Count total players
        total_query = db.query(func.count(func.distinct(Score.user_id))).filter(
            Score.dataset == dataset,
            Score.difficulty == difficulty
        )
        
        if since_date:
            total_query = total_query.filter(Score.created_at >= since_date)
        
        total_players = total_query.scalar() or 0
        
        return LeaderboardResponse(
            dataset=dataset,
            difficulty=difficulty,
            period=period,
            entries=entries,
            user_rank=user_rank,
            total_players=total_players
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get leaderboard: {str(e)}"
        )


@router.get("/user/stats")
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's game statistics."""
    try:
        if current_user.is_guest:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Stats not available for guest users"
            )
        
        # Get user's scores
        stats = db.query(
            func.count(Score.id).label('total_games'),
            func.sum(Score.total_score).label('total_score'),
            func.avg(Score.accuracy).label('avg_accuracy'),
            func.sum(func.cast(Score.beat_ai, db.bind.dialect.INTEGER)).label('beat_ai_count'),
            func.max(Score.total_score).label('best_score')
        ).filter(Score.user_id == current_user.id).first()
        
        # Get stats by dataset
        dataset_stats = db.query(
            Score.dataset,
            func.count(Score.id).label('games'),
            func.max(Score.total_score).label('best_score'),
            func.avg(Score.accuracy).label('avg_accuracy')
        ).filter(Score.user_id == current_user.id).group_by(Score.dataset).all()
        
        # Get recent games
        recent_games = db.query(Score).filter(
            Score.user_id == current_user.id
        ).order_by(desc(Score.created_at)).limit(10).all()
        
        return {
            "total_games": stats.total_games or 0,
            "total_score": stats.total_score or 0,
            "average_accuracy": float(stats.avg_accuracy) if stats.avg_accuracy else 0,
            "beat_ai_count": stats.beat_ai_count or 0,
            "best_score": stats.best_score or 0,
            "dataset_stats": [
                {
                    "dataset": ds.dataset,
                    "games": ds.games,
                    "best_score": ds.best_score,
                    "average_accuracy": float(ds.avg_accuracy)
                }
                for ds in dataset_stats
            ],
            "recent_games": [
                {
                    "dataset": game.dataset,
                    "difficulty": game.difficulty,
                    "score": game.total_score,
                    "accuracy": game.accuracy,
                    "beat_ai": game.beat_ai,
                    "created_at": game.created_at
                }
                for game in recent_games
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user stats: {str(e)}"
        )


def generate_explanation(user_answer: str, correct_answer: str, ai_answer: str, 
                        user_correct: bool, ai_confidence: float) -> str:
    """Generate explanation for the round result."""
    if user_correct:
        if ai_answer == correct_answer:
            return f"Correct! Both you and the AI got it right. The AI was {ai_confidence:.0%} confident it was '{correct_answer}'."
        else:
            return f"Excellent! You got it right with '{correct_answer}', but the AI guessed '{ai_answer}'. You beat the AI on this one!"
    else:
        if ai_answer == correct_answer:
            return f"Incorrect. You guessed '{user_answer}', but it was actually '{correct_answer}'. The AI got it right with {ai_confidence:.0%} confidence."
        else:
            return f"Incorrect. You guessed '{user_answer}' and the AI guessed '{ai_answer}', but it was actually '{correct_answer}'. Neither of you got it right!"