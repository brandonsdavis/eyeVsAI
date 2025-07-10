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

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON, 
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import os

from .database import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=True)
    username = Column(String(50), unique=True, nullable=True)
    display_name = Column(String(100), nullable=True)
    avatar_url = Column(Text, nullable=True)
    
    # Authentication
    password_hash = Column(String(255), nullable=True)  # NULL for social-only accounts
    email_verified = Column(Boolean, default=False)
    
    # Social OAuth IDs
    google_id = Column(String(255), unique=True, nullable=True)
    facebook_id = Column(String(255), unique=True, nullable=True)
    github_id = Column(String(255), unique=True, nullable=True)
    discord_id = Column(String(255), unique=True, nullable=True)
    twitter_id = Column(String(255), unique=True, nullable=True)
    apple_id = Column(String(255), unique=True, nullable=True)
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_guest = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    game_sessions = relationship("GameSession", back_populates="user")
    scores = relationship("Score", back_populates="user")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email', email),
        Index('idx_user_username', username),
        Index('idx_user_google_id', google_id),
        Index('idx_user_facebook_id', facebook_id),
        Index('idx_user_github_id', github_id),
        Index('idx_user_discord_id', discord_id),
        Index('idx_user_twitter_id', twitter_id),
        Index('idx_user_apple_id', apple_id),
        Index('idx_user_created_at', created_at),
    )


class GameSession(Base):
    __tablename__ = "game_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # NULL for guest users
    
    # Game configuration
    dataset = Column(String(50), nullable=False)  # pets, vegetables, etc.
    difficulty = Column(String(20), nullable=False)  # easy, medium, hard
    ai_model_key = Column(String(100), nullable=False)  # e.g., "transfer/resnet50"
    
    # Session state
    status = Column(String(20), default="active")  # active, completed, abandoned
    current_round = Column(Integer, default=1)
    total_rounds = Column(Integer, default=10)
    
    # Scoring
    correct_answers = Column(Integer, default=0)
    total_score = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="game_sessions")
    rounds = relationship("GameRound", back_populates="session")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_user_id', user_id),
        Index('idx_session_dataset', dataset),
        Index('idx_session_difficulty', difficulty),
        Index('idx_session_status', status),
        Index('idx_session_created_at', created_at),
    )


class GameRound(Base):
    __tablename__ = "game_rounds"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("game_sessions.id"), nullable=False)
    
    # Round details
    round_number = Column(Integer, nullable=False)
    image_path = Column(String(255), nullable=False)
    correct_answer = Column(String(100), nullable=False)
    
    # AI prediction commitment (for fairness)
    ai_prediction = Column(String(100), nullable=False)
    ai_confidence = Column(Float, nullable=False)
    ai_commitment_hash = Column(String(64), nullable=False)  # SHA-256 hash
    ai_commitment_salt = Column(String(64), nullable=False)  # Secret salt
    
    # User response
    user_answer = Column(String(100), nullable=True)  # NULL until answered
    user_correct = Column(Boolean, nullable=True)
    response_time_ms = Column(Integer, nullable=True)  # Time to answer in milliseconds
    
    # Scoring
    points_earned = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    answered_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    session = relationship("GameSession", back_populates="rounds")
    
    # Indexes
    __table_args__ = (
        Index('idx_round_session_id', session_id),
        Index('idx_round_number', round_number),
        Index('idx_round_created_at', created_at),
        UniqueConstraint('session_id', 'round_number'),
    )


class Score(Base):
    __tablename__ = "scores"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    session_id = Column(String(36), ForeignKey("game_sessions.id"), nullable=False)
    
    # Score details
    dataset = Column(String(50), nullable=False)
    difficulty = Column(String(20), nullable=False)
    ai_model_key = Column(String(100), nullable=False)
    
    # Performance metrics
    total_score = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    total_rounds = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=False)  # correct_answers / total_rounds
    average_response_time_ms = Column(Integer, nullable=True)
    
    # AI comparison
    ai_correct_answers = Column(Integer, nullable=False)
    ai_accuracy = Column(Float, nullable=False)
    beat_ai = Column(Boolean, nullable=False)  # user_accuracy > ai_accuracy
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="scores")
    session = relationship("GameSession")
    
    # Indexes
    __table_args__ = (
        Index('idx_score_user_id', user_id),
        Index('idx_score_dataset', dataset),
        Index('idx_score_difficulty', difficulty),
        Index('idx_score_total_score', total_score),
        Index('idx_score_accuracy', accuracy),
        Index('idx_score_created_at', created_at),
        Index('idx_score_leaderboard', dataset, difficulty, total_score),
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    token = Column(String(255), unique=True, nullable=False)
    
    # Token details
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_revoked = Column(Boolean, default=False)
    
    # Client info
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_refresh_token_user_id', user_id),
        Index('idx_refresh_token_token', token),
        Index('idx_refresh_token_expires_at', expires_at),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    
    # Event details
    event_type = Column(String(50), nullable=False)  # login, logout, game_start, etc.
    event_data = Column(JSON, nullable=True)
    
    # Request details
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_user_id', user_id),
        Index('idx_audit_event_type', event_type),
        Index('idx_audit_created_at', created_at),
    )