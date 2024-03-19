#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
"""IO utilities."""
import os
import fsspec


def exponential_backoff(attempt, base=2):
    return base**attempt


def get_creds():
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    out = {}
    out["key"] = aws_access_key_id
    out["secret"] = aws_secret_access_key
    return out


def fsspec_url_to_fs(url, **kwargs):
    if url.startswith("s3"):
        kwargs.update(get_creds())
    return fsspec.core.url_to_fs(url, **kwargs)


def list_timesplit_paths(url):
    """List split paths and expand paths if path/split/shard/xyz.tar."""
    fs, path = fsspec_url_to_fs(url)
    split_paths = fs.ls(path)
    split_paths_ex = []
    for spath in split_paths:
        sub_paths = fs.ls(spath)
        if not any(['.tar' for p in sub_paths]):
            split_paths_ex += sub_paths
        else:
            split_paths_ex += [spath]
    split_urls = [fs.protocol[0] + "://" + path for path in split_paths_ex]
    return split_urls
