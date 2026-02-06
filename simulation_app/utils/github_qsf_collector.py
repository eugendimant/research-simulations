"""
GitHub QSF File Collector

Automatically uploads new QSF files to a GitHub repository for collection purposes.
Runs silently in the background without interrupting user workflow.

Version: 1.2.0

Configuration via Streamlit secrets:
    GITHUB_TOKEN: Personal access token with repo write permissions
    GITHUB_REPO: Repository in format "owner/repo" (e.g., "eugendimant/research-simulations")
    GITHUB_QSF_PATH: Path within repo for QSF files (default: "simulation_app/example_files")
    GITHUB_COLLECTION_ENABLED: Set to "true" to enable (default: disabled)

Token Types Supported:
    - Classic tokens (ghp_XXXXXX): Require 'repo' scope
    - Fine-grained tokens (github_pat_XXXXXX): Require:
        * Repository access: Read and Write for Contents
        * Repository: Select the specific repo (e.g., eugendimant/research-simulations)

To generate a GitHub token:
    OPTION 1 - Classic Token (Recommended):
        1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
        2. Generate new token with 'repo' scope (full control of private repositories)
        3. Copy token (starts with 'ghp_') and add to Streamlit secrets as GITHUB_TOKEN

    OPTION 2 - Fine-grained Token:
        1. Go to GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens
        2. Generate new token with:
           - Resource owner: Your account
           - Repository access: Only select repositories → select your repo
           - Permissions → Repository permissions → Contents: Read and Write
        3. Copy token (starts with 'github_pat_') and add to Streamlit secrets as GITHUB_TOKEN
"""

import base64
import hashlib
import logging
import threading
from typing import Optional, Tuple
from functools import lru_cache

# Configure module logger
logger = logging.getLogger(__name__)

__version__ = "1.2.0"


def _validate_token_format(token: str) -> Tuple[bool, str]:
    """
    Validate GitHub token format.

    Returns:
        Tuple of (is_valid, message)
    """
    if not token:
        return False, "Token is empty"

    token = token.strip()

    # Classic tokens start with 'ghp_' and are ~40 chars
    if token.startswith("ghp_"):
        if len(token) >= 30:
            return True, "Classic token (ghp_) detected"
        return False, "Classic token appears too short"

    # Fine-grained tokens start with 'github_pat_'
    if token.startswith("github_pat_"):
        if len(token) >= 30:
            return True, "Fine-grained token (github_pat_) detected"
        return False, "Fine-grained token appears too short"

    # Old format tokens (deprecated but might still work)
    if len(token) == 40 and token.isalnum():
        return True, "Legacy token format detected (may be deprecated)"

    return False, f"Unknown token format. Token should start with 'ghp_' (classic) or 'github_pat_' (fine-grained). Got: {token[:10]}..."


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


def get_collection_status() -> dict:
    """
    Get detailed status of GitHub QSF collection configuration.

    Returns a dict with:
        - enabled: bool
        - token_valid: bool
        - token_message: str
        - repo: str
        - path: str
        - ready: bool (all checks pass)
    """
    config = _get_config()

    token = config.get("token", "")
    token_valid, token_message = _validate_token_format(token)

    return {
        "enabled": config.get("enabled", False),
        "token_valid": token_valid,
        "token_message": token_message,
        "repo": config.get("repo", ""),
        "path": config.get("path", ""),
        "ready": config.get("enabled", False) and token_valid and bool(config.get("repo")),
    }


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace problematic characters
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. ")
    sanitized = "".join(c if c in safe_chars else "_" for c in filename)
    # Collapse multiple underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    # Collapse multiple spaces
    while "  " in sanitized:
        sanitized = sanitized.replace("  ", " ")
    # Trim leading/trailing spaces and underscores
    sanitized = sanitized.strip(" _")
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

        # v1.2.0: Validate token format before attempting upload
        token = config.get('token', '')
        token_valid, token_msg = _validate_token_format(token)
        if not token_valid:
            logger.warning(f"GitHub token validation failed: {token_msg}")
            return False, f"Token validation failed: {token_msg}"

        headers = {
            "Authorization": f"token {token}",
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

        logger.info(f"Attempting GitHub upload: {filename} to {config['repo']}/{config['path']}")
        response = requests.put(url, headers=headers, json=payload, timeout=30)

        if response.status_code in (200, 201):
            logger.info(f"Successfully uploaded {filename} to GitHub (status: {response.status_code})")
            return True, f"Uploaded {filename}"
        else:
            # v1.2.0: More detailed error logging
            try:
                error_data = response.json()
                error_msg = error_data.get("message", "Unknown error")
                doc_url = error_data.get("documentation_url", "")
            except Exception:
                error_msg = f"HTTP {response.status_code}"
                doc_url = ""

            detailed_error = f"GitHub API error: {error_msg} (HTTP {response.status_code})"
            if response.status_code == 401:
                detailed_error += " - Token may be invalid or expired"
            elif response.status_code == 403:
                detailed_error += " - Token may lack required permissions (needs 'repo' scope or Contents write access)"
            elif response.status_code == 404:
                detailed_error += f" - Repository '{config['repo']}' or path '{config['path']}' not found"
            elif response.status_code == 422:
                detailed_error += " - File may already exist or content is invalid"

            logger.warning(detailed_error)
            if doc_url:
                logger.debug(f"GitHub docs: {doc_url}")
            return False, detailed_error

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
