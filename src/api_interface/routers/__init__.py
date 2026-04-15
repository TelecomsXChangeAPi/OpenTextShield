"""
API routers for OpenTextShield.

Includes both legacy and TMForum-compliant API endpoints.
"""

from . import health, metrics, prediction, feedback

# TMForum router is optional
try:
    from . import tmforum
except ImportError:
    tmforum = None