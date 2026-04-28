"""
pipeline/src/upload.py
───────────────────────
Upload raw documents and processed datasets to Azure Blob Storage.

Blob structure:
  financial-llm/
  ├── raw/
  │   ├── sec-10k/AAPL-10K-2023.json
  │   ├── sec-10q/AAPL-10Q-2023Q1.json
  │   └── earnings/AAPL-Q4-2023.json
  ├── processed/
  │   ├── train.jsonl
  │   ├── val.jsonl
  │   └── test.jsonl
  ├── checkpoints/
  │   └── mistral-finance-v1/
  └── models/
      └── mistral-finance-v1-merged/
"""

from __future__ import annotations

import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential
from loguru import logger
from tqdm import tqdm


CONTAINER_NAME = "financial-llm"


class AzureBlobManager:
    """Upload/download files to Azure Blob Storage."""

    def __init__(
        self,
        connection_string: str | None = None,
        account_url: str | None = None,
        container: str = CONTAINER_NAME,
    ):
        if connection_string:
            self.client = BlobServiceClient.from_connection_string(connection_string)
        elif account_url:
            credential = DefaultAzureCredential()
            self.client = BlobServiceClient(account_url=account_url, credential=credential)
        else:
            conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if conn:
                self.client = BlobServiceClient.from_connection_string(conn)
            else:
                raise ValueError("No Azure storage credentials provided")

        self.container = container
        self._ensure_container()

    def _ensure_container(self) -> None:
        try:
            self.client.create_container(self.container)
            logger.info(f"Created container: {self.container}")
        except Exception:
            pass  # Already exists

    def upload_file(
        self,
        local_path: Path,
        blob_path: str,
        overwrite: bool = False,
    ) -> str:
        """Upload a file to Azure Blob Storage. Returns the blob URL."""
        blob_client = self.client.get_blob_client(
            container=self.container,
            blob=blob_path,
        )
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=overwrite)
        url = blob_client.url
        logger.debug(f"Uploaded {local_path} → {url}")
        return url

    def upload_directory(
        self,
        local_dir: Path,
        blob_prefix: str,
        pattern: str = "**/*",
        overwrite: bool = False,
    ) -> list[str]:
        """Upload all files matching pattern from a local directory."""
        local_dir = Path(local_dir)
        files     = [f for f in local_dir.glob(pattern) if f.is_file()]
        urls      = []

        for f in tqdm(files, desc=f"Uploading to {blob_prefix}/"):
            rel_path = f.relative_to(local_dir)
            blob_path = f"{blob_prefix}/{rel_path}"
            url = self.upload_file(f, blob_path, overwrite=overwrite)
            urls.append(url)

        logger.info(f"Uploaded {len(urls)} files to {blob_prefix}/")
        return urls

    def download_file(self, blob_path: str, local_path: Path) -> None:
        """Download a blob to a local file."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob_client = self.client.get_blob_client(
            container=self.container, blob=blob_path
        )
        with open(local_path, "wb") as f:
            stream = blob_client.download_blob()
            stream.readinto(f)
        logger.debug(f"Downloaded {blob_path} → {local_path}")

    def download_dataset(self, split: str, local_dir: Path) -> Path:
        """Download train/val/test split from Azure."""
        blob_path  = f"processed/{split}.jsonl"
        local_path = Path(local_dir) / f"{split}.jsonl"
        self.download_file(blob_path, local_path)
        return local_path

    def list_blobs(self, prefix: str = "") -> list[str]:
        container_client = self.client.get_container_client(self.container)
        return [b.name for b in container_client.list_blobs(name_starts_with=prefix)]

    def upload_model_checkpoint(
        self,
        checkpoint_dir: Path,
        model_name: str,
        overwrite: bool = True,
    ) -> str:
        """Upload a full model checkpoint directory."""
        blob_prefix = f"checkpoints/{model_name}"
        urls = self.upload_directory(
            local_dir=checkpoint_dir,
            blob_prefix=blob_prefix,
            pattern="**/*",
            overwrite=overwrite,
        )
        logger.info(f"Uploaded {len(urls)} checkpoint files to {blob_prefix}")
        return blob_prefix
