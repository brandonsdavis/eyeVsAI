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

import os
import json
import pickle
import logging
from typing import Optional, Any, Union
from datetime import timedelta

try:
    import redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

logger = logging.getLogger(__name__)


class CacheService:
    """Service for caching using Redis or in-memory fallback."""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client: Optional[Redis] = None
        self.memory_cache = {}  # Fallback in-memory cache
        self._initialized = False
    
    async def initialize(self):
        """Initialize Redis connection."""
        if self._initialized:
            return
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = await Redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache.")
                self.redis_client = None
        else:
            logger.warning("Redis not available. Using in-memory cache.")
        
        self._initialized = True
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    # Try JSON first, then pickle
                    try:
                        return json.loads(value)
                    except:
                        try:
                            return pickle.loads(value)
                        except:
                            return value
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with optional TTL in seconds."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.redis_client:
                # Serialize value
                try:
                    serialized = json.dumps(value)
                except:
                    serialized = pickle.dumps(value)
                
                if ttl:
                    await self.redis_client.setex(key, ttl, serialized)
                else:
                    await self.redis_client.set(key, serialized)
            else:
                self.memory_cache[key] = value
                # Note: In-memory cache doesn't support TTL
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
    
    async def delete(self, key: str):
        """Delete value from cache."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.redis_client:
                await self.redis_client.delete(key)
            else:
                self.memory_cache.pop(key, None)
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.redis_client:
                return bool(await self.redis_client.exists(key))
            else:
                return key in self.memory_cache
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter in cache."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.redis_client:
                return await self.redis_client.incrby(key, amount)
            else:
                current = self.memory_cache.get(key, 0)
                new_value = current + amount
                self.memory_cache[key] = new_value
                return new_value
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return 0
    
    async def get_leaderboard(self, key: str, start: int = 0, end: int = -1) -> list:
        """Get sorted set (leaderboard) from cache."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.redis_client:
                # Get with scores in descending order
                results = await self.redis_client.zrevrange(
                    key, start, end, withscores=True
                )
                return [(member.decode() if isinstance(member, bytes) else member, score) 
                       for member, score in results]
            else:
                # Fallback: simple in-memory implementation
                leaderboard = self.memory_cache.get(key, {})
                sorted_items = sorted(
                    leaderboard.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                if end == -1:
                    return sorted_items[start:]
                else:
                    return sorted_items[start:end+1]
        except Exception as e:
            logger.error(f"Cache get_leaderboard error for key {key}: {e}")
            return []
    
    async def update_leaderboard(self, key: str, member: str, score: float):
        """Update score in sorted set (leaderboard)."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.redis_client:
                await self.redis_client.zadd(key, {member: score})
            else:
                # Fallback: simple in-memory implementation
                if key not in self.memory_cache:
                    self.memory_cache[key] = {}
                self.memory_cache[key][member] = score
        except Exception as e:
            logger.error(f"Cache update_leaderboard error for key {key}: {e}")
    
    async def cache_game_session(self, session_id: str, session_data: dict, ttl: int = 3600):
        """Cache game session data."""
        key = f"game_session:{session_id}"
        await self.set(key, session_data, ttl)
    
    async def get_game_session(self, session_id: str) -> Optional[dict]:
        """Get cached game session data."""
        key = f"game_session:{session_id}"
        return await self.get(key)
    
    async def cache_model_prediction(self, model_key: str, image_hash: str, 
                                   prediction: dict, ttl: int = 86400):
        """Cache model prediction result."""
        key = f"prediction:{model_key}:{image_hash}"
        await self.set(key, prediction, ttl)
    
    async def get_model_prediction(self, model_key: str, image_hash: str) -> Optional[dict]:
        """Get cached model prediction."""
        key = f"prediction:{model_key}:{image_hash}"
        return await self.get(key)
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
        self.memory_cache.clear()
        self._initialized = False


# Global cache service instance
cache_service = CacheService()