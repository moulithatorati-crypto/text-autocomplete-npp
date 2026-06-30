"""
Logging utilities for the NPP system.
Provides structured logging across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class LoggerSetup:
    """Configure and manage loggers for the NPP system."""
    
    _loggers = {}
    
    @staticmethod
    def setup_logger(
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        format_str: Optional[str] = None
    ) -> logging.Logger:
        """
        Setup a logger with both console and optional file handlers.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file
            format_str: Optional custom format string
            
        Returns:
            Configured logger instance
        """
        if name in LoggerSetup._loggers:
            return LoggerSetup._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Default format
        if format_str is None:
            format_str = (
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        formatter = logging.Formatter(format_str)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        LoggerSetup._loggers[name] = logger
        return logger
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get an existing logger.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name in LoggerSetup._loggers:
            return LoggerSetup._loggers[name]
        return logging.getLogger(name)


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Convenience function to get or create a logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    return LoggerSetup.setup_logger(name, level, log_file)