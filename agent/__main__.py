#!/usr/bin/env python3
"""
Unified CLI for Factor Optimization Agents
==========================================

Supports both Claude (Anthropic) and Gemini (Google) optimization agents.

Usage:
    # Claude optimization (recommended)
    python -m agent --claude --factor VLUE --max-iterations 30
    
    # Gemini optimization (legacy) 
    python -m agent --gemini --smart VLUE --max-iterations 50
    
    # Interactive mode
    python -m agent --claude --interactive
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import FACTORS


def main():
    """CLI entry point for optimization agents."""
    parser = argparse.ArgumentParser(
        description="Factor Timing Optimization Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Claude optimization (recommended)
  python -m agent --claude --factor VLUE --max-iterations 30
  
  # Gemini optimization (legacy)
  python -m agent --gemini --factor VLUE --iterate --min-iterations 20
  
  # Interactive mode with Claude
  python -m agent --claude --interactive
  
  # Smart optimization with Claude
  python -m agent --claude --smart VLUE --max-iterations 50
        """
    )
    
    # Agent selection
    agent_group = parser.add_mutually_exclusive_group(required=True)
    agent_group.add_argument("--claude", action="store_true", help="Use Claude (Anthropic) agent")
    agent_group.add_argument("--gemini", action="store_true", help="Use Gemini (Google) agent")
    
    # Common options
    parser.add_argument("--factor", choices=FACTORS, help="Factor to optimize")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Claude options
    claude_group = parser.add_argument_group("Claude options")
    claude_group.add_argument("--max-iterations", type=int, default=30, help="Max iterations for Claude")
    claude_group.add_argument("--target-sharpe", type=float, default=0.5, help="Target Sharpe ratio")
    claude_group.add_argument("--claude-model", default="claude-3-5-sonnet-20241022", help="Claude model")
    claude_group.add_argument("--target-type", choices=["premium", "ratio"], default="premium",
                             help="'premium' (factor - benchmark) or 'ratio' (long factor/short benchmark pair trade)")
    claude_group.add_argument("--benchmark", default="SPY", help="Benchmark symbol for ratio mode (default: SPY)")
    
    # Gemini options (legacy)
    gemini_group = parser.add_argument_group("Gemini options")
    gemini_group.add_argument("--iterate", action="store_true", help="Iterative optimization")
    gemini_group.add_argument("--smart", metavar="FACTOR", help="Smart optimization for factor")
    gemini_group.add_argument("--min-iterations", type=int, default=10, help="Minimum iterations for Gemini")
    gemini_group.add_argument("--gemini-model", default="gemini-3-pro-preview", help="Gemini model")
    
    args = parser.parse_args()
    
    # Initialize the appropriate agent
    if args.claude:
        try:
            from agent.claude_agent import ClaudeOptimizationAgent
            agent = ClaudeOptimizationAgent(
                model=args.claude_model,
                verbose=args.verbose,
                target_type=args.target_type,
                benchmark=args.benchmark
            )
        except ValueError as e:
            print(f"❌ Claude setup error: {e}")
            print("Set ANTHROPIC_API_KEY environment variable")
            return 1
        except ImportError:
            print("❌ Claude not available. Install with: pip install anthropic")
            return 1
        
        if args.interactive:
            agent.interactive_mode()
        elif args.factor:
            agent.optimize_factor(
                factor=args.factor,
                max_iterations=args.max_iterations,
                target_sharpe=args.target_sharpe
            )
        else:
            print("❌ Specify --factor or --interactive for Claude agent")
            return 1
    
    elif args.gemini:
        try:
            from agent.heuristic_agent import HeuristicOptimizationAgent
            agent = HeuristicOptimizationAgent(
                model=args.gemini_model,
                verbose=args.verbose
            )
        except ValueError as e:
            print(f"❌ Gemini setup error: {e}")
            print("Set GOOGLE_API_KEY environment variable")
            return 1
        except ImportError:
            print("❌ Gemini not available. Install with: pip install google-generativeai")
            return 1
        
        if args.interactive:
            agent.interactive_mode()
        elif args.smart:
            factor = args.smart.upper()
            if factor not in FACTORS:
                print(f"❌ Invalid factor. Choose from: {FACTORS}")
                return 1
            agent.smart_optimize(
                factor=factor,
                max_iterations=args.max_iterations,
                use_bayesian=True,
                use_genetic=True,
                use_reflection=True
            )
        elif args.iterate and args.factor:
            agent.iterative_optimize(
                factor=args.factor,
                min_iterations=args.min_iterations
            )
        else:
            print("❌ Specify optimization mode for Gemini agent")
            print("Use --smart <factor>, --iterate --factor <factor>, or --interactive")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

