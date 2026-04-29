# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
from contextlib import contextmanager
from typing import Any, Generator, Union
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystemBase


class S3FileSystem(FileSystemBase):
    """Implementation of FileSystem for AWS S3 storage."""

    def __init__(self, credential_path: str) -> None:
        with open(credential_path, "r") as f:
            config = json.load(f)
        self.s3_client = boto3.client("s3", **config)

    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        """
        Create a stream to read from or write to S3.

        Args:
            path: S3 URI in the format s3://bucket-name/key
            mode: 'rb' for reading, 'wb' for writing
        """
        path_str = str(path)
        bucket, key = self._parse_s3_uri(path_str)

        if mode == "rb":
            # For reading, download to memory stream
            stream = io.BytesIO()
            try:
                self.s3_client.download_fileobj(bucket, key, stream)
                stream.seek(0)
                yield stream
            finally:
                stream.close()
        elif mode == "wb":
            # For writing, use memory stream then upload
            stream = io.BytesIO()
            try:
                yield stream
                stream.seek(0)
                self.s3_client.upload_fileobj(stream, bucket, key)
            finally:
                stream.close()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]:
        """Concatenate S3 path with suffix."""
        path_str = str(path)
        if path_str.endswith("/"):
            return f"{path_str}{suffix}"
        return f"{path_str}/{suffix}"

    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        """
        Rename (or move) an object in S3.

        In S3, this is implemented as a copy followed by a deletion of the original.
        """
        src_bucket, src_key = self._parse_s3_uri(str(path))
        dst_bucket, dst_key = self._parse_s3_uri(str(new_path))

        # Copy the object
        copy_source = {"Bucket": src_bucket, "Key": src_key}
        self.s3_client.copy(copy_source, dst_bucket, dst_key)

        # Delete the original
        self.s3_client.delete_object(Bucket=src_bucket, Key=src_key)

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """Initialize and validate S3 path."""
        path_str = str(path)
        if not path_str.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {path_str}. It must start with 's3://'")
        return path_str

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        """
        Create a "directory" in S3.

        Note: S3 doesn't have real directories, but we can create an empty object
        with a trailing slash to simulate a directory.
        """
        path_str = str(path)
        if not path_str.endswith("/"):
            path_str += "/"

        bucket, key = self._parse_s3_uri(path_str)
        if key:  # Don't create empty object if this is just a bucket
            self.s3_client.put_object(Bucket=bucket, Key=key)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """Validate if the checkpoint_id is a valid S3 URI."""
        checkpoint_id_str = str(checkpoint_id)
        try:
            if not checkpoint_id_str.startswith("s3://"):
                return False
            parsed = urlparse(checkpoint_id_str)
            return bool(parsed.netloc and parsed.path)  # Must have bucket and key
        except Exception:
            return False

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        """Check if an object exists in S3."""
        bucket, key = self._parse_s3_uri(str(path))
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                return False
            raise  # Re-raise other errors

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        """Remove a file from S3."""
        bucket, key = self._parse_s3_uri(str(path))
        self.s3_client.delete_object(Bucket=bucket, Key=key)

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """
        Parse an S3 URI into bucket and key.

        Args:
            uri: S3 URI in the format s3://bucket-name/key

        Returns:
            Tuple of (bucket_name, key)

        Raises:
            ValueError: If the URI is invalid
        """
        uri = uri if isinstance(uri, str) else str(uri)
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")

        parsed = urlparse(uri)
        bucket = parsed.netloc

        # Remove leading slash from key
        key = parsed.path.lstrip("/")

        if not bucket:
            raise ValueError(f"Invalid S3 URI: {uri}. No bucket specified")

        return bucket, key

    def list_files_recursive(self, s3_dir: Union[str, os.PathLike]) -> list[str]:
        """List all files in a directory in S3."""
        bucket, prefix = self._parse_s3_uri(str(s3_dir).removesuffix("/"))
        prefix = prefix.removesuffix("/")
        scan_prefix = f"{prefix}/" if prefix else ""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=scan_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") and obj.get("Size", 0) == 0:
                    continue
                suffix = key[len(scan_prefix) :] if scan_prefix else key
                if suffix:
                    out.append(suffix)
        return sorted(out)

    def download_to_local(
        self, s3_uri: Union[str, os.PathLike], local_path: Union[str, os.PathLike]
    ) -> None:
        """Download a file from S3 to local."""
        bucket, key = self._parse_s3_uri(str(s3_uri))
        local_path_str = str(local_path)
        os.makedirs(os.path.dirname(local_path_str), exist_ok=True)
        self.s3_client.download_file(bucket, key, local_path_str)

    def head_object(
        self, s3_uri: Union[str, os.PathLike], checksum_mode: bool = False
    ) -> dict[str, Any]:
        """Get the metadata of a file in S3."""
        bucket, key = self._parse_s3_uri(str(s3_uri))
        kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if checksum_mode:
            kwargs["ChecksumMode"] = "ENABLED"
        return self.s3_client.head_object(**kwargs)

    def close(self) -> None:
        """Close the S3 client."""
        self.s3_client.close()


class S3StorageWriter(FileSystemWriter):
    def __init__(self, credential_path: str, path: str, **kwargs) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            credential_path (str): The path to the credential file of accessing AWS S3.
            path (str): The S3 URI to write checkpoints to.
            kwargs (dict): Keyword arguments to pass to the parent :class:`FileSystemWriter`.
        """
        super().__init__(path=path, sync_files=False, **kwargs)
        self.fs = S3FileSystem(credential_path)
        self.path = self.fs.init_path(path)


class S3StorageReader(FileSystemReader):
    def __init__(self, credential_path: str, path: Union[str, os.PathLike]) -> None:
        """
        Initialize an S3 reader for distributed checkpointing.

        Args:
            credential_path (str): The path to the credential file of accessing AWS S3.
            path (Union[str, os.PathLike]): The S3 URI to read checkpoints from.
        """
        super().__init__(path=path)
        self.fs = S3FileSystem(credential_path)
        self.path = self.fs.init_path(path)
        # self.sync_files = False


if __name__ == "__main__":
    import time

    credential_path = "credentials/s3_checkpoint.secret"
    s3_uri = "s3://flashdreams/assets/"
    local_path = "/tmp/flashdreams/assets/"
    fs = S3FileSystem(credential_path)
    paths = fs.list_files_recursive(s3_uri)
    for path in paths:
        print(path)

    for path in paths:
        print("downloading", path)
        tic = time.time()
        fs.download_to_local(os.path.join(s3_uri, path), os.path.join(local_path, path))
        toc = time.time()
        print("downloaded", path, "in", toc - tic, "seconds")
        break
