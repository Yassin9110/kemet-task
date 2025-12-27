"""
Logging utilities for the Multilingual RAG Ingestion Pipeline.

This module provides a custom logger with:
- Formatted console output with colors
- Optional file logging
- Stage-aware logging with timing
- Progress indicators
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List


class LogColor:
    """ANSI color codes for terminal output."""
    
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    @classmethod
    def disable(cls) -> None:
        """Disable all colors (for non-TTY output)."""
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.BLACK = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""
        cls.BRIGHT_RED = ""
        cls.BRIGHT_GREEN = ""
        cls.BRIGHT_YELLOW = ""
        cls.BRIGHT_BLUE = ""
        cls.BRIGHT_MAGENTA = ""
        cls.BRIGHT_CYAN = ""
        cls.BRIGHT_WHITE = ""


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors based on log level."""
    
    LEVEL_COLORS = {
        logging.DEBUG: LogColor.DIM + LogColor.WHITE,
        logging.INFO: LogColor.BRIGHT_WHITE,
        logging.WARNING: LogColor.BRIGHT_YELLOW,
        logging.ERROR: LogColor.BRIGHT_RED,
        logging.CRITICAL: LogColor.BOLD + LogColor.BRIGHT_RED,
    }
    
    LEVEL_ICONS = {
        logging.DEBUG: "○",
        logging.INFO: "●",
        logging.WARNING: "⚠",
        logging.ERROR: "✗",
        logging.CRITICAL: "✗",
    }
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize the formatter.
        
        Args:
            use_colors: Whether to use ANSI colors.
        """
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and icons."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        
        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, LogColor.WHITE)
            icon = self.LEVEL_ICONS.get(record.levelno, "●")
            
            formatted = (
                f"{LogColor.DIM}{timestamp}{LogColor.RESET} "
                f"{color}{icon} {record.getMessage()}{LogColor.RESET}"
            )
        else:
            level_name = record.levelname
            formatted = f"{timestamp} | {level_name:<8} | {record.getMessage()}"
        
        return formatted


class FileFormatter(logging.Formatter):
    """Plain formatter for file output."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record without colors."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level_name = record.levelname
        return f"{timestamp} | {level_name:<8} | {record.getMessage()}"


@dataclass
class StageTimer:
    """Timer for tracking stage duration."""
    
    stage_name: str
    stage_number: int
    total_stages: int
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def duration_ms(self) -> int:
        """Get duration in milliseconds."""
        end = self.end_time or time.time()
        return int((end - self.start_time) * 1000)
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.duration_ms / 1000.0
    
    def stop(self) -> int:
        """Stop the timer and return duration in ms."""
        self.end_time = time.time()
        return self.duration_ms


@dataclass
class PipelineTimer:
    """Timer for tracking entire pipeline duration."""
    
    document_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    stage_durations: Dict[str, int] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> int:
        """Get total duration in milliseconds."""
        end = self.end_time or time.time()
        return int((end - self.start_time) * 1000)
    
    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        return self.duration_ms / 1000.0
    
    def record_stage(self, stage_name: str, duration_ms: int) -> None:
        """Record a stage's duration."""
        self.stage_durations[stage_name] = duration_ms
    
    def stop(self) -> int:
        """Stop the timer and return total duration in ms."""
        self.end_time = time.time()
        return self.duration_ms


class PipelineLogger:
    """
    Custom logger for the ingestion pipeline.
    
    Provides formatted output with colors, stage tracking,
    timing information, and optional file logging.
    """
    
    SEPARATOR = "═" * 60
    THIN_SEPARATOR = "─" * 60
    
    def __init__(
        self,
        name: str = "pipeline",
        log_level: str = "INFO",
        log_to_file: bool = False,
        log_dir: str = "./logs",
        log_filename: str = "pipeline.log",
        use_colors: bool = True
    ):
        """
        Initialize the pipeline logger.
        
        Args:
            name: Logger name.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_to_file: Whether to write logs to file.
            log_dir: Directory for log files.
            log_filename: Log file name.
            use_colors: Whether to use ANSI colors in console.
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_to_file = log_to_file
        self.log_dir = Path(log_dir)
        self.log_filename = log_filename
        self.use_colors = use_colors and sys.stdout.isatty()
        
        # Disable colors if not TTY
        if not self.use_colors:
            LogColor.disable()
        
        # Create logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.log_level)
        self._logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(ColoredFormatter(use_colors=self.use_colors))
        self._logger.addHandler(console_handler)
        
        # File handler (optional)
        if self.log_to_file:
            self._setup_file_handler()
        
        # State tracking
        self._current_document_id: Optional[str] = None
        self._pipeline_timer: Optional[PipelineTimer] = None
        self._current_stage_timer: Optional[StageTimer] = None
        self._warnings: List[str] = []
    
    def _setup_file_handler(self) -> None:
        """Set up file logging handler."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / self.log_filename
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(FileFormatter())
        self._logger.addHandler(file_handler)
    
    @property
    def log_file_path(self) -> Optional[Path]:
        """Get the path to the log file if file logging is enabled."""
        if self.log_to_file:
            return self.log_dir / self.log_filename
        return None
    
    # -------------------------------------------------------------------------
    # Basic Logging Methods
    # -------------------------------------------------------------------------
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self._logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self._logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._logger.warning(message)
        self._warnings.append(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log a critical message."""
        self._logger.critical(message)
    
    # -------------------------------------------------------------------------
    # Visual Separators
    # -------------------------------------------------------------------------
    
    def separator(self, thick: bool = True) -> None:
        """Print a separator line."""
        sep = self.SEPARATOR if thick else self.THIN_SEPARATOR
        if self.use_colors:
            self._logger.info(f"{LogColor.DIM}{sep}{LogColor.RESET}")
        else:
            self._logger.info(sep)
    
    def blank_line(self) -> None:
        """Print a blank line."""
        self._logger.info("")
    
    def header(self, text: str) -> None:
        """Print a header with separators."""
        self.separator(thick=True)
        if self.use_colors:
            self._logger.info(f"{LogColor.BOLD}{LogColor.BRIGHT_CYAN}{text}{LogColor.RESET}")
        else:
            self._logger.info(text)
        self.separator(thick=True)
    
    def subheader(self, text: str) -> None:
        """Print a subheader with thin separators."""
        self.separator(thick=False)
        if self.use_colors:
            self._logger.info(f"{LogColor.CYAN}{text}{LogColor.RESET}")
        else:
            self._logger.info(text)
        self.separator(thick=False)
    
    # -------------------------------------------------------------------------
    # Pipeline Lifecycle
    # -------------------------------------------------------------------------
    
    def start_pipeline(self, document_id: str, file_path: str) -> None:
        """
        Log the start of pipeline processing.
        
        Args:
            document_id: The document ID being processed.
            file_path: Path to the source file.
        """
        self._current_document_id = document_id
        self._pipeline_timer = PipelineTimer(document_id=document_id)
        self._warnings = []
        
        self.header("Starting Ingestion Pipeline")
        self.info(f"Document ID: {document_id}")
        self.info(f"File: {file_path}")
        self.separator(thick=True)
    
    def end_pipeline_success(
        self,
        parent_count: int,
        child_count: int,
        edge_count: int,
        total_tokens: int
    ) -> int:
        """
        Log successful pipeline completion.
        
        Args:
            parent_count: Number of parent chunks created.
            child_count: Number of child chunks created.
            edge_count: Number of edges created.
            total_tokens: Total tokens processed.
            
        Returns:
            Total processing duration in milliseconds.
        """
        duration_ms = 0
        if self._pipeline_timer:
            duration_ms = self._pipeline_timer.stop()
        
        self.separator(thick=True)
        
        if self.use_colors:
            self._logger.info(
                f"{LogColor.BOLD}{LogColor.BRIGHT_GREEN}✓ Ingestion Complete{LogColor.RESET}"
            )
        else:
            self._logger.info("✓ Ingestion Complete")
        
        self.info(f"  Document ID: {self._current_document_id}")
        self.info(f"  Parents: {parent_count}")
        self.info(f"  Children: {child_count}")
        self.info(f"  Edges: {edge_count}")
        self.info(f"  Total Tokens: {total_tokens}")
        self.info(f"  Duration: {duration_ms / 1000:.2f}s")
        
        if self._warnings:
            self.info(f"  Warnings: {len(self._warnings)}")
            for warning in self._warnings:
                self.info(f"    - {warning}")
        
        self.separator(thick=True)
        
        self._current_document_id = None
        self._pipeline_timer = None
        
        return duration_ms
    
    def end_pipeline_failure(self, error_message: str) -> int:
        """
        Log pipeline failure.
        
        Args:
            error_message: Description of the failure.
            
        Returns:
            Processing duration before failure in milliseconds.
        """
        duration_ms = 0
        if self._pipeline_timer:
            duration_ms = self._pipeline_timer.stop()
        
        self.separator(thick=True)
        
        if self.use_colors:
            self._logger.info(
                f"{LogColor.BOLD}{LogColor.BRIGHT_RED}✗ Ingestion Failed{LogColor.RESET}"
            )
        else:
            self._logger.info("✗ Ingestion Failed")
        
        self.info(f"  Document ID: {self._current_document_id}")
        self.error(f"  Error: {error_message}")
        self.info(f"  Duration: {duration_ms / 1000:.2f}s")
        
        self.separator(thick=True)
        
        self._current_document_id = None
        self._pipeline_timer = None
        
        return duration_ms
    
    # -------------------------------------------------------------------------
    # Stage Lifecycle
    # -------------------------------------------------------------------------
    
    def start_stage(self, stage_number: int, stage_name: str, total_stages: int = 11) -> None:
        """
        Log the start of a pipeline stage.
        
        Args:
            stage_number: Stage number (1-indexed).
            stage_name: Name of the stage.
            total_stages: Total number of stages.
        """
        self._current_stage_timer = StageTimer(
            stage_name=stage_name,
            stage_number=stage_number,
            total_stages=total_stages
        )
        
        if self.use_colors:
            self._logger.info(
                f"{LogColor.BRIGHT_BLUE}[Stage {stage_number}/{total_stages}]{LogColor.RESET} "
                f"{LogColor.BOLD}{stage_name}{LogColor.RESET} ..."
            )
        else:
            self._logger.info(f"[Stage {stage_number}/{total_stages}] {stage_name} ...")
    
    def end_stage_success(self, details: Optional[str] = None) -> int:
        """
        Log successful stage completion.
        
        Args:
            details: Optional details about the stage result.
            
        Returns:
            Stage duration in milliseconds.
        """
        duration_ms = 0
        stage_name = "Unknown"
        stage_number = 0
        total_stages = 0
        
        if self._current_stage_timer:
            duration_ms = self._current_stage_timer.stop()
            stage_name = self._current_stage_timer.stage_name
            stage_number = self._current_stage_timer.stage_number
            total_stages = self._current_stage_timer.total_stages
            
            if self._pipeline_timer:
                self._pipeline_timer.record_stage(stage_name, duration_ms)
        
        if self.use_colors:
            status = f"{LogColor.BRIGHT_GREEN}✓{LogColor.RESET}"
            timing = f"{LogColor.DIM}({duration_ms}ms){LogColor.RESET}"
            self._logger.info(
                f"{LogColor.BRIGHT_BLUE}[Stage {stage_number}/{total_stages}]{LogColor.RESET} "
                f"{stage_name} {status} {timing}"
            )
        else:
            self._logger.info(
                f"[Stage {stage_number}/{total_stages}] {stage_name} ✓ ({duration_ms}ms)"
            )
        
        if details:
            self.info(f"  → {details}")
        
        self._current_stage_timer = None
        return duration_ms
    
    def end_stage_failure(self, error_message: str) -> int:
        """
        Log stage failure.
        
        Args:
            error_message: Description of the failure.
            
        Returns:
            Stage duration before failure in milliseconds.
        """
        duration_ms = 0
        stage_name = "Unknown"
        stage_number = 0
        total_stages = 0
        
        if self._current_stage_timer:
            duration_ms = self._current_stage_timer.stop()
            stage_name = self._current_stage_timer.stage_name
            stage_number = self._current_stage_timer.stage_number
            total_stages = self._current_stage_timer.total_stages
        
        if self.use_colors:
            status = f"{LogColor.BRIGHT_RED}✗{LogColor.RESET}"
            timing = f"{LogColor.DIM}({duration_ms}ms){LogColor.RESET}"
            self._logger.info(
                f"{LogColor.BRIGHT_BLUE}[Stage {stage_number}/{total_stages}]{LogColor.RESET} "
                f"{stage_name} {status} {timing}"
            )
        else:
            self._logger.info(
                f"[Stage {stage_number}/{total_stages}] {stage_name} ✗ ({duration_ms}ms)"
            )
        
        self.error(f"  → {error_message}")
        
        self._current_stage_timer = None
        return duration_ms
    
    def stage_progress(self, message: str) -> None:
        """
        Log progress within a stage.
        
        Args:
            message: Progress message.
        """
        if self.use_colors:
            self._logger.info(f"  {LogColor.DIM}→{LogColor.RESET} {message}")
        else:
            self._logger.info(f"  → {message}")
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def log_dict(self, title: str, data: Dict[str, Any], indent: int = 2) -> None:
        """
        Log a dictionary with formatting.
        
        Args:
            title: Title for the dictionary.
            data: Dictionary to log.
            indent: Number of spaces for indentation.
        """
        self.info(f"{title}:")
        prefix = " " * indent
        for key, value in data.items():
            self.info(f"{prefix}{key}: {value}")
    
    def log_list(self, title: str, items: List[Any], indent: int = 2) -> None:
        """
        Log a list with formatting.
        
        Args:
            title: Title for the list.
            items: List to log.
            indent: Number of spaces for indentation.
        """
        self.info(f"{title}:")
        prefix = " " * indent
        for item in items:
            self.info(f"{prefix}- {item}")
    
    def get_warnings(self) -> List[str]:
        """Get all warnings logged during current pipeline run."""
        return self._warnings.copy()
    
    def clear_warnings(self) -> None:
        """Clear the warnings list."""
        self._warnings = []


def get_logger(
    name: str = "pipeline",
    log_level: str = "INFO",
    log_to_file: bool = False,
    log_dir: str = "./logs",
    log_filename: str = "pipeline.log"
) -> PipelineLogger:
    """
    Get or create a pipeline logger.
    
    Args:
        name: Logger name.
        log_level: Logging level.
        log_to_file: Whether to write logs to file.
        log_dir: Directory for log files.
        log_filename: Log file name.
        
    Returns:
        A PipelineLogger instance.
    """
    return PipelineLogger(
        name=name,
        log_level=log_level,
        log_to_file=log_to_file,
        log_dir=log_dir,
        log_filename=log_filename
    )


def create_logger_from_config(config: Any) -> PipelineLogger:
    """
    Create a logger from a PipelineConfig object.
    
    Args:
        config: PipelineConfig instance.
        
    Returns:
        A configured PipelineLogger instance.
    """
    return PipelineLogger(
        name="pipeline",
        log_level=config.log_level,
        log_to_file=config.log_to_file,
        log_dir=config.log_dir,
        log_filename=config.log_filename
    )