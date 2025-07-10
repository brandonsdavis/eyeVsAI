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
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import hashlib
import mimetypes

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None
    ClientError = Exception

logger = logging.getLogger(__name__)


class S3Service:
    """Service for managing S3 storage for models and images."""
    
    def __init__(self):
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.model_bucket = os.getenv("S3_BUCKET_NAME", "eyevsai-models")
        self.image_bucket = os.getenv("S3_BUCKET_IMAGES", "eyevsai-images")
        self.s3_client = None
        self._initialized = False
        self.model_cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/eyevsai_models"))
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize S3 client."""
        if self._initialized:
            return
        
        if not S3_AVAILABLE:
            logger.warning("boto3 not available. S3 features disabled.")
            self._initialized = True
            return
        
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            logger.warning("AWS credentials not configured. S3 features disabled.")
            self._initialized = True
            return
        
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.model_bucket)
            logger.info("S3 service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None
        
        self._initialized = True
    
    async def upload_model(self, local_path: Path, model_key: str) -> bool:
        """Upload a model to S3."""
        if not self._initialized:
            await self.initialize()
        
        if not self.s3_client:
            logger.warning("S3 not available. Skipping model upload.")
            return False
        
        try:
            # Create S3 key structure: models/{model_type}/{variation}/{version}/
            s3_key = f"models/{model_key}"
            
            # Upload directory recursively
            if local_path.is_dir():
                for file_path in local_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        file_key = f"{s3_key}/{relative_path}"
                        
                        logger.info(f"Uploading {file_path} to s3://{self.model_bucket}/{file_key}")
                        self.s3_client.upload_file(
                            str(file_path),
                            self.model_bucket,
                            file_key
                        )
            else:
                # Single file upload
                file_key = f"{s3_key}/{local_path.name}"
                logger.info(f"Uploading {local_path} to s3://{self.model_bucket}/{file_key}")
                self.s3_client.upload_file(
                    str(local_path),
                    self.model_bucket,
                    file_key
                )
            
            # Upload metadata
            metadata = {
                "model_key": model_key,
                "upload_timestamp": os.path.getmtime(str(local_path)),
                "source_path": str(local_path)
            }
            
            metadata_key = f"{s3_key}/metadata.json"
            self.s3_client.put_object(
                Bucket=self.model_bucket,
                Key=metadata_key,
                Body=json.dumps(metadata),
                ContentType="application/json"
            )
            
            logger.info(f"Successfully uploaded model {model_key} to S3")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload model {model_key}: {e}")
            return False
    
    async def download_model(self, model_key: str) -> Optional[Path]:
        """Download a model from S3 to local cache."""
        if not self._initialized:
            await self.initialize()
        
        if not self.s3_client:
            logger.warning("S3 not available. Cannot download model.")
            return None
        
        try:
            # Check if model is already cached
            local_model_path = self.model_cache_dir / model_key
            if local_model_path.exists():
                logger.info(f"Model {model_key} already cached locally")
                return local_model_path
            
            # Create local directory
            local_model_path.mkdir(parents=True, exist_ok=True)
            
            # List and download all files for this model
            s3_prefix = f"models/{model_key}/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.model_bucket, Prefix=s3_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        relative_path = s3_key[len(s3_prefix):]
                        
                        if relative_path:  # Skip directory markers
                            local_file_path = local_model_path / relative_path
                            local_file_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            logger.info(f"Downloading s3://{self.model_bucket}/{s3_key}")
                            self.s3_client.download_file(
                                self.model_bucket,
                                s3_key,
                                str(local_file_path)
                            )
            
            logger.info(f"Successfully downloaded model {model_key} from S3")
            return local_model_path
            
        except Exception as e:
            logger.error(f"Failed to download model {model_key}: {e}")
            return None
    
    async def upload_image(self, image_path: Path, dataset: str, class_name: str) -> Optional[str]:
        """Upload an image to S3 and return its URL."""
        if not self._initialized:
            await self.initialize()
        
        if not self.s3_client:
            logger.warning("S3 not available. Cannot upload image.")
            return None
        
        try:
            # Generate S3 key
            image_hash = self._hash_file(image_path)
            s3_key = f"images/{dataset}/{class_name}/{image_hash}{image_path.suffix}"
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(str(image_path))
            if not content_type:
                content_type = "image/jpeg"
            
            # Upload image
            self.s3_client.upload_file(
                str(image_path),
                self.image_bucket,
                s3_key,
                ExtraArgs={'ContentType': content_type}
            )
            
            # Generate URL
            url = f"https://{self.image_bucket}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
            
            logger.info(f"Successfully uploaded image to {url}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to upload image {image_path}: {e}")
            return None
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models available in S3."""
        if not self._initialized:
            await self.initialize()
        
        if not self.s3_client:
            return []
        
        try:
            models = []
            
            # List model directories
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(
                Bucket=self.model_bucket,
                Prefix="models/",
                Delimiter="/"
            ):
                if 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        model_prefix = prefix['Prefix']
                        
                        # Try to get metadata
                        metadata_key = f"{model_prefix}metadata.json"
                        try:
                            response = self.s3_client.get_object(
                                Bucket=self.model_bucket,
                                Key=metadata_key
                            )
                            metadata = json.loads(response['Body'].read())
                            
                            models.append({
                                "model_key": metadata.get("model_key", model_prefix.strip("/")),
                                "s3_prefix": model_prefix,
                                "metadata": metadata
                            })
                        except:
                            # No metadata, just add the prefix
                            models.append({
                                "model_key": model_prefix.strip("/").replace("models/", ""),
                                "s3_prefix": model_prefix,
                                "metadata": {}
                            })
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def sync_models_from_s3(self) -> bool:
        """Sync model registry with S3 bucket."""
        if not self._initialized:
            await self.initialize()
        
        if not self.s3_client:
            return False
        
        try:
            # Get list of models from S3
            s3_models = await self.list_models()
            
            # Update local registry
            registry_path = self.model_cache_dir / "s3_model_registry.json"
            
            registry = {
                "models": s3_models,
                "last_sync": datetime.utcnow().isoformat()
            }
            
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            
            logger.info(f"Synced {len(s3_models)} models from S3")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync models from S3: {e}")
            return False
    
    def _hash_file(self, file_path: Path) -> str:
        """Generate hash of file content."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    
    async def cleanup(self):
        """Cleanup resources."""
        self._initialized = False


# Global S3 service instance
s3_service = S3Service()