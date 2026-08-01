"""
Outbound-fetch guard for URLs that originate outside this codebase.

The web tier accepts a ``preview_url`` straight off the request body
(``POST /api/place`` → ``atlas.resolve_and_embed``), so an unguarded
``requests.get`` on it lets any caller aim the server at localhost, the LAN,
or a cloud metadata endpoint. Every fetch of a *caller-influenced* URL must
go through ``safe_get`` instead.

Two independent controls, so neither alone has to be perfect:

1. **Host allowlist** (the primary control). Preview audio only ever comes
   from three CDNs — Spotify ``p.scdn.co``, Deezer ``*.dzcdn.net``, Apple
   ``audio-ssl.itunes.apple.com`` / ``*.mzstatic.com``. Anything else is
   refused before a socket is opened. This also closes DNS-rebinding, which
   would otherwise need control of one of those zones.
2. **Public-IP check** (defense in depth). Every address the hostname
   resolves to must be globally routable, so an allowlisted name that
   resolves into a private range still cannot be reached.

Redirects are followed manually (``allow_redirects=False``) so each hop is
re-validated — a 302 into ``169.254.169.254`` is the classic bypass of a
check applied only to the initial URL.

Failures raise ``UnsafeURLError``. Callers must surface a generic message to
the client: the distinction between "refused", "connection refused" and
"404" is itself the oracle that turns blind SSRF into a working internal
port scanner, and the byte-length of a rejected response leaks more still.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

# Suffix match: a value with a leading dot matches any subdomain, a bare
# hostname matches exactly. Override with ANTHER_ALLOWED_FETCH_HOSTS
# (comma-separated) when a CDN moves; default is closed.
DEFAULT_ALLOWED_HOSTS = (
    "p.scdn.co",
    ".scdn.co",
    ".dzcdn.net",
    ".itunes.apple.com",
    ".mzstatic.com",
)

MAX_REDIRECTS = 3
MAX_BYTES = 32 * 1024 * 1024      # a preview clip is ~1-2 MB; this is slack


class UnsafeURLError(Exception):
    """URL failed the allowlist / public-IP / scheme checks."""


def allowed_hosts() -> tuple[str, ...]:
    raw = os.environ.get("ANTHER_ALLOWED_FETCH_HOSTS", "").strip()
    if not raw:
        return DEFAULT_ALLOWED_HOSTS
    return tuple(h.strip().lower() for h in raw.split(",") if h.strip())


def _host_allowed(host: str) -> bool:
    host = host.lower().rstrip(".")
    for entry in allowed_hosts():
        if entry.startswith("."):
            if host.endswith(entry) and len(host) > len(entry):
                return True
        elif host == entry:
            return True
    return False


def _resolves_public(host: str) -> bool:
    """True only if every A/AAAA record for ``host`` is globally routable.
    Fails closed when resolution fails."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        # is_global excludes loopback, link-local (169.254.0.0/16 metadata),
        # private, reserved, multicast and unspecified in one check.
        if not ip.is_global:
            return False
    return True


def validate_url(url: str) -> str:
    """Return ``url`` if it is safe to fetch, else raise ``UnsafeURLError``."""
    if not url or not isinstance(url, str):
        raise UnsafeURLError("empty url")
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise UnsafeURLError(f"scheme not allowed: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("no host in url")
    if not _host_allowed(host):
        raise UnsafeURLError(f"host not on allowlist: {host!r}")
    if not _resolves_public(host):
        raise UnsafeURLError(f"host does not resolve to a public address: {host!r}")
    return url


def safe_get(url: str, timeout: int = 20) -> requests.Response:
    """
    Validated GET for a caller-influenced URL.

    Follows up to ``MAX_REDIRECTS`` hops, re-validating each ``Location``
    against the same rules, and caps the response body at ``MAX_BYTES`` so a
    hostile (or merely huge) endpoint can't exhaust memory.
    """
    current = validate_url(url)
    for _ in range(MAX_REDIRECTS + 1):
        resp = requests.get(current, timeout=timeout, allow_redirects=False,
                            stream=True)
        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("Location", "")
            resp.close()
            if not location:
                raise UnsafeURLError("redirect without Location")
            # Relative redirects stay on an already-validated host, but
            # re-validating the joined URL is cheaper than reasoning about it.
            current = validate_url(requests.compat.urljoin(current, location))
            continue

        try:
            chunks = []
            size = 0
            for chunk in resp.iter_content(64 * 1024):
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_BYTES:
                    raise UnsafeURLError("response exceeds size cap")
            body = b"".join(chunks)
        finally:
            resp.close()
        # _content is what .content returns; set it so callers using the
        # normal Response API see the fully-buffered body.
        resp._content = body
        return resp

    raise UnsafeURLError("too many redirects")
