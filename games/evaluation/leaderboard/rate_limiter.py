# -*- coding: utf-8 -*-
"""Rate limiter for API calls to prevent overwhelming the API service."""
import time
import asyncio
import threading
from typing import Optional


class RateLimiter:
    """Thread-safe rate limiter for API calls.
    
    Ensures a minimum interval between API calls to prevent rate limiting.
    Supports both sync and async calls.
    """
    
    def __init__(self, min_interval: float = 0.0):
        """Initialize rate limiter.
        
        Args:
            min_interval: Minimum seconds between API calls. If 0.0, no rate limiting.
        """
        self.min_interval = min_interval
        self._last_call_time = 0.0
        self._lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to maintain minimum interval between calls (sync version)."""
        if self.min_interval <= 0.0:
            return
        
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time
            
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)
            
            self._last_call_time = time.time()
    
    async def async_wait_if_needed(self):
        """Wait if necessary to maintain minimum interval between calls (async version)."""
        if self.min_interval <= 0.0:
            return
        
        # Calculate sleep time while holding lock, then release lock before await
        # This prevents deadlock: we can't hold a threading.Lock() while awaiting
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time
            sleep_time = max(0.0, self.min_interval - time_since_last_call)
        
        # Sleep outside the lock to avoid deadlock
        if sleep_time > 0.0:
            await asyncio.sleep(sleep_time)
        
        # Update last call time after sleep
        with self._lock:
            self._last_call_time = time.time()


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def set_global_rate_limiter(min_interval: float):
    """Set the global rate limiter interval.
    
    Args:
        min_interval: Minimum seconds between API calls. If 0.0, no rate limiting.
    """
    global _global_rate_limiter
    _global_rate_limiter = RateLimiter(min_interval)


def get_global_rate_limiter() -> Optional[RateLimiter]:
    """Get the global rate limiter instance.
    
    Returns:
        Rate limiter instance or None if not set.
    """
    return _global_rate_limiter


def apply_rate_limiting_to_openai_model():
    """Apply rate limiting to OpenAIChatModel by monkey patching its __call__ method.
    
    This wraps the model's __call__ method to add rate limiting before each API call.
    Only modifies the behavior if a global rate limiter is set.
    OpenAIChatModel uses async __call__, so we wrap it with async rate limiting.
    """
    global _global_rate_limiter
    if _global_rate_limiter is None or _global_rate_limiter.min_interval <= 0.0:
        return
    
    try:
        from agentscope.model import OpenAIChatModel
        
        # Store original __call__ method if not already wrapped
        if not hasattr(OpenAIChatModel, '_original___call__'):
            OpenAIChatModel._original___call__ = OpenAIChatModel.__call__
        
        # Create wrapped async __call__ method
        async def rate_limited_call(self, *args, **kwargs):
            """Wrapped async __call__ method with rate limiting."""
            await _global_rate_limiter.async_wait_if_needed()
            try:
                return await self._original___call__(*args, **kwargs)
            except UnicodeEncodeError as e:
                # Re-raise with more context about the rate limiter
                raise UnicodeEncodeError(
                    e.encoding,
                    e.object,
                    e.start,
                    e.end,
                    f"{e.reason} (occurred during API call with rate limiting enabled)"
                ) from e
        
        # Replace __call__ method
        OpenAIChatModel.__call__ = rate_limited_call
        
    except ImportError:
        # If OpenAIChatModel is not available, silently skip
        pass

