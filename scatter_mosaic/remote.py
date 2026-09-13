"""Read a remote file in pieces, so a 500 MB parquet costs a few MB.

A columnar file keeps an index of where every column of every row group lives.
Given random access a reader can fetch only the bytes it actually wants — and
HTTP range requests are random access. That is the whole trick here: this turns
a URL into something `pyarrow.parquet` can seek around in, which is what lets
`fetch_example.py` pull tens of thousands of rows of attributes, and a few dozen
images, out of a five-gigabyte dataset without downloading it.

The server has to answer with `206 Partial Content`. HuggingFace's parquet
endpoints and Google Cloud Storage both do. They also rate-limit, hence the
retry policy and the reused connection.
"""

from __future__ import annotations

import io
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "scatter-mosaic/1.0"


def session(retries: int = 6) -> requests.Session:
    """One keep-alive connection that backs off instead of giving up on a 429."""
    policy = Retry(
        total=retries, backoff_factor=1.5,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    made = requests.Session()
    made.headers["User-Agent"] = USER_AGENT
    made.mount("https://", HTTPAdapter(max_retries=policy, pool_maxsize=4))
    return made


class RemoteFile(io.RawIOBase):
    """A seekable, read-only file over HTTP range requests."""

    def __init__(self, url: str, connection: requests.Session | None = None, timeout: int = 120,
                 pace: float = 0.05) -> None:
        self.url, self.timeout, self.pos, self.fetched, self.requests = url, timeout, 0, 0, 0
        # Hundreds of range requests in a burst is what gets a host to answer 429.
        self.pace, self._last = pace, 0.0
        self.session = connection or session()
        head = self.session.head(url, allow_redirects=True, timeout=timeout)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = (offset if whence == io.SEEK_SET
                    else self.pos + offset if whence == io.SEEK_CUR
                    else self.size + offset)
        return self.pos

    def read(self, size: int = -1) -> bytes:
        wanted = self.size - self.pos if size is None or size < 0 else min(size, self.size - self.pos)
        if wanted <= 0:
            return b""
        gap = self.pace - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        answer = self.session.get(
            self.url, timeout=self.timeout,
            headers={"Range": f"bytes={self.pos}-{self.pos + wanted - 1}"},
        )
        answer.raise_for_status()
        chunk = answer.content
        self._last = time.monotonic()
        self.pos += len(chunk)
        self.fetched += len(chunk)
        self.requests += 1
        return chunk

    def readinto(self, buffer) -> int:  # noqa: ANN001 - matches the io signature
        chunk = self.read(len(buffer))
        buffer[:len(chunk)] = chunk
        return len(chunk)

    @property
    def megabytes(self) -> float:
        return self.fetched / 1_048_576
