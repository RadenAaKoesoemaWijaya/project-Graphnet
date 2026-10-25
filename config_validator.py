"""
Configuration validation module for ASTINA
Validates all configuration settings at application startup
"""
import os
import logging
from config import (
    MAX_FILE_SIZE, MAX_CSV_FILE_SIZE, MAX_PARQUET_FILE_SIZE, MAX_EXCEL_FILE_SIZE,
    DATABASE_CONFIG, LLM_COPILOT_CONFIG
)

logger = logging.getLogger(__name__)

def validate_config() -> tuple[bool, list[str]]:
    """
    Validate all configuration settings at startup
    
    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    
    # Check file size consistency
    if MAX_FILE_SIZE < max(MAX_CSV_FILE_SIZE, MAX_PARQUET_FILE_SIZE, MAX_EXCEL_FILE_SIZE):
        errors.append(
            f"MAX_FILE_SIZE ({MAX_FILE_SIZE}) must be >= individual format limits "
            f"(CSV: {MAX_CSV_FILE_SIZE}, Parquet: {MAX_PARQUET_FILE_SIZE}, Excel: {MAX_EXCEL_FILE_SIZE})"
        )
    
    # Check database config if enabled
    db_enabled = os.getenv("ENABLE_DATABASE", "0").lower() in ("1", "true", "yes")
    if db_enabled:
        if not DATABASE_CONFIG.get('password'):
            errors.append("DB_PASSWORD environment variable must be set when database is enabled")
        if not DATABASE_CONFIG.get('username'):
            errors.append("DB_USERNAME environment variable must be set when database is enabled")
    
    # Check LLM config
    llm_provider = os.getenv("LLM_PROVIDER", "heuristic")
    valid_providers = ["heuristic", "gemini", "openai", "ollama"]
    if llm_provider not in valid_providers:
        errors.append(f"Invalid LLM_PROVIDER '{llm_provider}'. Must be one of: {valid_providers}")
    
    # Check API keys for non-heuristic providers
    if llm_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        errors.append("GEMINI_API_KEY must be set when LLM_PROVIDER is 'gemini'")
    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY must be set when LLM_PROVIDER is 'openai'")
    
    # Check memory limits
    memory_limit_mb = int(os.getenv("ASTINA_MEMORY_LIMIT_MB", "4096"))
    if memory_limit_mb < 1024:
        errors.append("ASTINA_MEMORY_LIMIT_MB must be at least 1024 (1GB)")
    
    is_valid = len(errors) == 0
    
    if not is_valid:
        logger.error(f"Configuration validation failed with {len(errors)} errors:")
        for error in errors:
            logger.error(f"  - {error}")
    else:
        logger.info("Configuration validation passed")
    
    return is_valid, errors


def validate_environment() -> tuple[bool, list[str]]:
    """
    Validate environment variables and system requirements
    
    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    
    # Check required directories
    required_dirs = ["cache", "models", "logs", "plots", "studies"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
                logger.info(f"Created directory: {dir_name}")
            except Exception as e:
                errors.append(f"Failed to create directory '{dir_name}': {e}")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 11):
        errors.append(f"Python 3.11+ required, current version: {sys.version}")
    
    # Check disk space (at least 5GB free)
    import shutil
    try:
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 5:
            errors.append(f"Insufficient disk space: {free_gb:.2f}GB free, 5GB required")
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
    
    is_valid = len(errors) == 0
    
    if not is_valid:
        logger.error(f"Environment validation failed with {len(errors)} errors:")
        for error in errors:
            logger.error(f"  - {error}")
    else:
        logger.info("Environment validation passed")
    
    return is_valid, errors


def run_all_validations() -> tuple[bool, list[str]]:
    """
    Run all validations (config and environment)
    
    Returns:
        (is_valid, list of all error messages)
    """
    all_errors = []
    
    config_valid, config_errors = validate_config()
    all_errors.extend(config_errors)
    
    env_valid, env_errors = validate_environment()
    all_errors.extend(env_errors)
    
    is_valid = config_valid and env_valid
    
    return is_valid, all_errors
