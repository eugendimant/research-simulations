"""
GitHub QSF File Collector

Automatically uploads new QSF files to a GitHub repository for collection purposes.
Runs silently in the background without interrupting user workflow.

Version: 1.0.0

Configuration via Streamlit secrets:
    GITHUB_TOKEN: Personal access token with repo write permissions
    GITHUB_REPO: Repository in format "owner/repo" (e.g., "eugendimant/research-simulations")
    GITHUB_QSF_PATH: Path within repo for QSF files (default: "simulation_app/example_files")
    GITHUB_COLLECTION_ENABLED: Set to "true" to enable (default: disabled)

To generate a GitHub token:
    1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
    2. Generate new token with 'repo' scope
    3. Copy token and add to Streamlit secrets as GITHUB_TOKEN
"""

import base64
import hashlib
import logging
import threading
from typing import Optional, Tuple
from functools import lru_cache

# Configure module logger
logger = logging.getLogger(__name__)

__version__ = "1.0.0"


def _get_config() -> dict:
    """Get GitHub configuration from Streamlit secrets."""
    try:
        import streamlit as st
        return {
            "token": st.secrets.get("GITHUB_TOKEN", ""),
            "repo": st.secrets.get("GITHUB_REPO", "eugendimant/research-simulations"),
            "path": st.secrets.get("GITHUB_QSF_PATH", "simulation_app/example_files"),
            "enabled": str(st.secrets.get("GITHUB_COLLECTION_ENABLED", "false")).lower() == "true",
        }
    except Exception:
        return {"token": "", "repo": "", "path": "", "enabled": False}


def is_collection_enabled() -> bool:
    """Check if QSF collection is enabled and properly configured."""
    config = _get_config()
    return config["enabled"] and bool(config["token"]) and bool(config["repo"])


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace problematic characters
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. ")
    sanitized = "".join(c if c in safe_chars else "_" for c in filename)
    # Collapse multiple underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    # Ensure .qsf extension
    if not sanitized.lower().endswith(".qsf"):
        sanitized = sanitized.rstrip(".") + ".qsf"
    return sanitized.strip()


def _file_exists_in_repo(filename: str, config: dict) -> bool:
    """Check if a file with this name already exists in the GitHub repo."""
    try:
        import requests

        headers = {
            "Authorization": f"token {config['token']}",
            "Accept": "application/vnd.github.v3+json",
        }

        # GitHub API: Get contents of directory
        url = f"https://api.github.com/repos/{config['repo']}/contents/{config['path']}"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            files = response.json()
            existing_names = {f["name"].lower() for f in files if isinstance(f, dict)}
            return filename.lower() in existing_names
        elif response.status_code == 404:
            # Directory doesn't exist yet - file definitely doesn't exist
            return False
        else:
            logger.warning(f"GitHub API returned {response.status_code} when checking for file")
            return True  # Assume exists to avoid duplicates on error

    except Exception as e:
        logger.warning(f"Error checking if file exists in GitHub: {e}")
        return True  # Assume exists on error to be safe


def _upload_to_github(filename: str, content: bytes, config: dict) -> Tuple[bool, str]:
    """Upload a file to GitHub repository."""
    try:
        import requests

        headers = {
            "Authorization": f"token {config['token']}",
            "Accept": "application/vnd.github.v3+json",
        }

        # Encode content as base64
        content_b64 = base64.b64encode(content).decode("utf-8")

        # GitHub API: Create or update file
        file_path = f"{config['path']}/{filename}"
        url = f"https://api.github.com/repos/{config['repo']}/contents/{file_path}"

        # Create commit message with content hash for traceability
        content_hash = hashlib.sha256(content).hexdigest()[:8]
        commit_message = f"Auto-collect QSF: {filename} [{content_hash}]"

        payload = {
            "message": commit_message,
            "content": content_b64,
            "branch": "main",  # Or could be configurable
        }

        response = requests.put(url, headers=headers, json=payload, timeout=30)

        if response.status_code in (200, 201):
            logger.info(f"Successfully uploaded {filename} to GitHub")
            return True, f"Uploaded {filename}"
        else:
            error_msg = response.json().get("message", "Unknown error")
            logger.warning(f"GitHub upload failed: {error_msg}")
            return False, error_msg

    except Exception as e:
        logger.warning(f"Error uploading to GitHub: {e}")
        return False, str(e)


def collect_qsf_async(filename: str, content: bytes) -> None:
    """
    Asynchronously collect a QSF file to GitHub.

    This function runs in a background thread to avoid blocking the UI.
    It silently succeeds or fails without interrupting the user.

    Args:
        filename: Original filename of the QSF file
        content: Raw bytes of the QSF file content
    """
    def _background_upload():
        try:
            config = _get_config()

            if not config["enabled"]:
                return

            if not config["token"]:
                logger.debug("QSF collection enabled but no GitHub token configured")
                return

            # Sanitize filename
            safe_filename = _sanitize_filename(filename)

            # Check if file already exists
            if _file_exists_in_repo(safe_filename, config):
                logger.debug(f"QSF file {safe_filename} already exists in repo, skipping")
                return

            # Upload to GitHub
            success, message = _upload_to_github(safe_filename, content, config)

            if success:
                logger.info(f"QSF collection: {message}")
            else:
                logger.debug(f"QSF collection skipped: {message}")

        except Exception as e:
            # Never let collection errors affect the main app
            logger.debug(f"QSF collection error (non-fatal): {e}")

    # Run in background thread
    thread = threading.Thread(target=_background_upload, daemon=True)
    thread.start()


def collect_qsf_sync(filename: str, content: bytes) -> Tuple[bool, str]:
    """
    Synchronously collect a QSF file to GitHub.

    Use this for testing or when you need to confirm the upload completed.

    Args:
        filename: Original filename of the QSF file
        content: Raw bytes of the QSF file content

    Returns:
        Tuple of (success: bool, message: str)
    """
    config = _get_config()

    if not config["enabled"]:
        return False, "QSF collection is not enabled"

    if not config["token"]:
        return False, "GitHub token not configured"

    # Sanitize filename
    safe_filename = _sanitize_filename(filename)

    # Check if file already exists
    if _file_exists_in_repo(safe_filename, config):
        return False, f"File {safe_filename} already exists in repository"

    # Upload to GitHub
    return _upload_to_github(safe_filename, content, config)
