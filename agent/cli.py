#!/usr/bin/env python3
"""
Heuristic Optimization Agent CLI
=================================

Command-line interface for the heuristic optimization agent.

Usage:
    python -m agent                              # Interactive mode
    python -m agent --analyze VLUE               # Analyze a specific factor
    python -m agent --optimize VLUE              # Optimize a factor
    python -m agent --compare-all                # Compare all factors
    python -m agent --query "your question"      # Single query mode
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Heuristic Optimization Agent - Improve factor timing strategies using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agent                        # Start interactive chat
  python -m agent --analyze VLUE         # Analyze Value factor strategy
  python -m agent --optimize SIZE        # Optimize Size factor weights
  python -m agent --compare-all          # Compare all factor strategies
  python -m agent -q "What features predict VLUE best?"
        """,
    )
    
    parser.add_argument(
        "--analyze", "-a",
        metavar="FACTOR",
        choices=["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
        help="Analyze a specific factor's heuristic model",
    )
    
    parser.add_argument(
        "--optimize", "-o",
        metavar="FACTOR", 
        choices=["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
        help="Optimize signal weights for a factor (single pass)",
    )
    
    parser.add_argument(
        "--iterate", "-i",
        metavar="FACTOR",
        choices=["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
        help="Iterative optimization until convergence",
    )
    
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=20,
        help="Minimum iterations before checking convergence (default: 20)",
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum optimization iterations (default: 50)",
    )
    
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="Stop if Sharpe improvement < this after min-iterations (default: 0.05)",
    )
    
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Don't apply best config to code (just report)",
    )
    
    parser.add_argument(
        "--smart",
        type=str,
        metavar="FACTOR",
        help="Run SMART optimization with Bayesian, Genetic, Reflection, OOS validation",
    )
    
    parser.add_argument(
        "--compare-all", "-c",
        action="store_true",
        help="Compare performance across all factors",
    )
    
    parser.add_argument(
        "--query", "-q",
        metavar="QUESTION",
        help="Single query mode - ask a question and exit",
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output (only show final response)",
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save session to file when done",
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it with: export GOOGLE_API_KEY=your-key")
        sys.exit(1)
    
    # Import agent (after path setup)
    from agent.heuristic_agent import HeuristicOptimizationAgent
    
    # Create agent
    agent = HeuristicOptimizationAgent(
        model=args.model,
        verbose=not args.quiet,
    )
    
    print("="*60)
    print("🤖 Heuristic Optimization Agent (Gemini)")
    print("="*60)
    print(f"Model: {args.model}")
    print()
    
    try:
        if args.analyze:
            print(f"Analyzing {args.analyze} factor strategy...")
            response = agent.analyze_factor(args.analyze)
            print("\n" + response)
        
        elif args.optimize:
            print(f"Optimizing {args.optimize} factor strategy...")
            response = agent.optimize_factor(args.optimize)
            print("\n" + response)
        
        elif args.smart:
            print(f"🧠 SMART optimization for {args.smart}...")
            print(f"   Max iterations: {args.max_iterations}")
            print(f"   Strategies: Bayesian + Genetic + Reflection + OOS Validation")
            print()
            
            result = agent.smart_optimize(
                factor=args.smart,
                max_iterations=args.max_iterations,
                use_bayesian=True,
                use_genetic=True,
                use_reflection=True,
                validate_oos=True,
                build_ensemble=True,
                apply_to_code=not args.no_apply,
            )
            
            print(f"\n{'='*60}")
            print("SMART OPTIMIZATION RESULTS")
            print(f"{'='*60}")
            print(f"Factor: {result['factor']}")
            print(f"Baseline Sharpe: {result['baseline_sharpe']:.3f}")
            print(f"Best Sharpe: {result['best_sharpe']:.3f}")
            print(f"Improvement: {result['improvement']:+.3f}")
            print(f"Best iteration: {result['best_iteration']}")
            
            if result.get('ensemble_tested'):
                print(f"✅ Ensemble tested")
            if result.get('oos_validated'):
                print(f"✅ Out-of-sample validated")
            
            if result['best_config']:
                print(f"\nBest configuration:")
                import json
                print(json.dumps(result['best_config'], indent=2))
        
        elif args.iterate:
            print(f"🚀 Iterative optimization for {args.iterate}...")
            print(f"   Min iterations: {args.min_iterations}")
            print(f"   Max iterations: {args.max_iterations}")
            print(f"   Min improvement threshold: {args.min_improvement}")
            print()
            
            result = agent.iterative_optimize(
                factor=args.iterate,
                min_iterations=args.min_iterations,
                max_iterations=args.max_iterations,
                min_improvement=args.min_improvement,
                apply_to_code=not args.no_apply,
            )
            
            print(f"\n{'='*60}")
            print("FINAL RESULTS")
            print(f"{'='*60}")
            print(f"Factor: {result['factor']}")
            print(f"Baseline Sharpe: {result['baseline_sharpe']:.3f}")
            print(f"Best Sharpe: {result['best_sharpe']:.3f}")
            print(f"Improvement: {result['improvement']:+.3f}")
            print(f"Best iteration: {result['best_iteration']}")
            
            if result['best_config']:
                print(f"\nBest configuration:")
                import json
                print(json.dumps(result['best_config'], indent=2))
        
        elif args.compare_all:
            print("Comparing all factor strategies...")
            response = agent.compare_all_factors()
            print("\n" + response)
        
        elif args.query:
            response = agent.chat(args.query)
            print("\n" + response)
        
        else:
            # Interactive mode
            print("Interactive mode - type 'quit' or 'exit' to end")
            print("Commands: 'analyze <FACTOR>', 'optimize <FACTOR>', 'iterate <FACTOR>', 'compare', 'reset', 'save'")
            print("-"*60)
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle special commands
                lower_input = user_input.lower()
                
                if lower_input in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break
                
                elif lower_input == "reset":
                    agent.reset()
                    continue
                
                elif lower_input == "save":
                    agent.save_session()
                    continue
                
                elif lower_input == "compare":
                    response = agent.compare_all_factors()
                    print("\n" + response)
                    continue
                
                elif lower_input.startswith("analyze "):
                    factor = lower_input.replace("analyze ", "").upper().strip()
                    if factor in ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]:
                        response = agent.analyze_factor(factor)
                        print("\n" + response)
                    else:
                        print(f"Unknown factor: {factor}. Use VLUE, SIZE, QUAL, USMV, or MTUM")
                    continue
                
                elif lower_input.startswith("optimize "):
                    factor = lower_input.replace("optimize ", "").upper().strip()
                    if factor in ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]:
                        response = agent.optimize_factor(factor)
                        print("\n" + response)
                    else:
                        print(f"Unknown factor: {factor}. Use VLUE, SIZE, QUAL, USMV, or MTUM")
                    continue
                
                elif lower_input.startswith("iterate "):
                    factor = lower_input.replace("iterate ", "").upper().strip()
                    if factor in ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]:
                        result = agent.iterative_optimize(factor)
                        print(f"\n✅ Best Sharpe: {result['best_sharpe']:.3f} (improvement: {result['improvement']:+.3f})")
                    else:
                        print(f"Unknown factor: {factor}. Use VLUE, SIZE, QUAL, USMV, or MTUM")
                    continue
                
                # Regular chat
                response = agent.chat(user_input)
                print("\n" + response)
        
        # Save session if requested
        if args.save:
            agent.save_session()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
