"""
Heuristic Optimization Agent
============================

AI agents for analyzing and improving factor timing strategies.
Supports both Claude (Anthropic) and Gemini (Google) models.
"""

from . import tools
from . import strategies

# Import agents conditionally to avoid dependency errors
_available_agents = []

try:
    from .claude_agent import ClaudeOptimizationAgent
    _available_agents.append('ClaudeOptimizationAgent')
except ImportError:
    ClaudeOptimizationAgent = None

try:
    from .heuristic_agent import HeuristicOptimizationAgent
    _available_agents.append('HeuristicOptimizationAgent')
except ImportError:
    HeuristicOptimizationAgent = None

__version__ = "0.2.0"
__all__ = ['tools', 'strategies'] + _available_agents

