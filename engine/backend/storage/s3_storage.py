"""
backend/storage/s3_storage.py — AWS S3 Storage backend.

Activate via:
    SNAPPY_STORAGE=s3
    SNAPPY_S3_BUCKET=your-bucket
    SNAPPY_S3_REGION=us-east-1
    SNAPPY_S3_PUBLIC_BASE=https://cdn.example.com   # optional CloudFront
    AWS_ACCESS_KEY_ID=…
    AWS_SECRET_ACCESS_KEY=…

Photos are written under {sid}/{filename} keys. Public URL is either:
  - the signed S3 URL (default, expires in 7 days)
  - or {SNAPPY_S3_PUBLIC_BASE}/{sid}/{filename} if you've fronted the
    bucket with CloudFront and want immutable URLs.

Requires `pip install boto3`. Falls back gracefully if boto3 missing.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from .base import Storage, StorageObject

log = logging.getLogger("snappy.storage.s3")


class S3Storage:
    name = "s3"

    def __init__(self, bucket: str, region: str,
                 public_base: Optional[str] = None,
                 prefix: str = ""):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise RuntimeError(
                f"boto3 not installed for S3Storage: {e}. "
                "Install with: pip install boto3"
            )
        self._bucket = bucket
        self._region = region
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._public_base = (public_base.rstrip("/") if public_base else None)

        # Reasonable retries; S3 is generally rock-solid but transient
        # 503s happen when you hit prefix-shard limits during bursts.
        self._client = boto3.client(
            "s3",
            region_name=region,
            config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
        )
        log.info(f"S3Storage ready: bucket={bucket} region={region} "
                 f"public_base={public_base or '(signed URLs)'}")

    def _key(self, sid: str, filename: str) -> str:
        return f"{self._prefix}{sid}/{filename}"

    def put_photo(self, sid: str, filename: str, data: bytes) -> StorageObject:
        key = self._key(sid, filename)
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data,
            ContentType="image/jpeg", CacheControl="public, max-age=2592000",
        )
        url = (f"{self._public_base}/{key}" if self._public_base
               else self.signed_url(sid, filename))
        return StorageObject(url=url, storage_path=f"s3://{self._bucket}/{key}",
                             bytes=len(data))

    def get_photo(self, sid: str, filename: str) -> Optional[bytes]:
        try:
            r = self._client.get_object(Bucket=self._bucket, Key=self._key(sid, filename))
            return r["Body"].read()
        except Exception as e:
            log.debug(f"S3 get {sid}/{filename}: {e}")
            return None

    def signed_url(self, sid: str, filename: str,
                   expires_seconds: int = 7 * 24 * 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": self._key(sid, filename)},
            ExpiresIn=int(expires_seconds),
        )

    def delete_session(self, sid: str) -> int:
        prefix = self._key(sid, "")
        n = 0
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if not objs:
                continue
            self._client.delete_objects(Bucket=self._bucket,
                                        Delete={"Objects": objs, "Quiet": True})
            n += len(objs)
        return n
