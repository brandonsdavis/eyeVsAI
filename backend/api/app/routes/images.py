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

from fastapi import APIRouter, HTTPException, Path as PathParam
from fastapi.responses import FileResponse
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])


@router.get("/{image_path:path}")
async def serve_image(image_path: str = PathParam(..., description="Path to the image file")):
    """Serve image files for the game."""
    try:
        # Security: Prevent directory traversal attacks
        if ".." in image_path or image_path.startswith("/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid image path"
            )
        
        # Convert to Path object
        full_path = Path(image_path)
        
        # Check if it's an absolute path (from our datasets)
        if full_path.is_absolute() and full_path.exists():
            file_path = full_path
        else:
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            file_path = project_root / image_path
        
        # Validate file exists and is an image
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        if not file_path.is_file():
            raise HTTPException(
                status_code=400,
                detail="Path is not a file"
            )
        
        # Check file extension
        allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        if file_path.suffix.lower() not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type"
            )
        
        # Determine media type
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp"
        }
        
        media_type = media_types.get(file_path.suffix.lower(), "image/jpeg")
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "X-Content-Type-Options": "nosniff"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve image {image_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to serve image"
        )