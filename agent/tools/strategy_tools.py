"""
Strategy Tools
==============

Tools that expose the advanced optimization strategies to the agent.
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .strategy_tools import ContinuousRefinementController
import json

from ..strategies import (
    Config,
    OptimizationState,
    BayesianOptimizer,
    GeneticOptimizer,
    EnsembleBuilder,
    SelfReflection,
    OutOfSampleValidator,
    ExplorationPhase,
)
import random
from .backtest import run_backtest


# Global state for optimization session
_optimization_state: Optional[OptimizationState] = None
_bayesian_optimizer: Optional[BayesianOptimizer] = None
_genetic_optimizer: Optional[GeneticOptimizer] = None
_continuous_mode: bool = False
_refinement_controller: Optional['ContinuousRefinementController'] = None


class ContinuousRefinementController:
    """Autonomous controller for continuous strategy refinement"""
    
    def __init__(self, target_sharpe: float = 1.5, max_stagnation: int = 10):
        self.target_sharpe = target_sharpe
        self.max_stagnation = max_stagnation
        self.refinement_history = []
        self.last_action = None
        self.meta_learning_insights = {}
        
    def should_continue_refinement(self, state: OptimizationState) -> bool:
        """Decide if refinement should continue"""
        if not state.configs:
            return True
            
        # Stop if target reached
        if state.best_sharpe >= self.target_sharpe:
            return False
            
        # Stop if too much stagnation
        if state.stagnation_count >= self.max_stagnation:
            # Try one more meta-learning intervention
            if self.last_action != "meta_intervention":
                return True
            return False
            
        return True
    
    def get_next_action(self, state: OptimizationState) -> Dict[str, Any]:
        """Determine next autonomous action"""
        convergence = state.get_convergence_signal()
        
        # High-level autonomous decisions
        if state.stagnation_count >= 5:
            # Stagnation detected - meta intervention
            action = self._plan_meta_intervention(state)
            self.last_action = "meta_intervention"
        elif convergence > 0.9 and len(state.configs) > 20:
            # High convergence - focus on exploitation
            action = self._plan_exploitation_phase(state)
            self.last_action = "exploitation"
        elif len(state.configs) < 10:
            # Early phase - broad exploration
            action = self._plan_exploration_phase(state)
            self.last_action = "exploration"
        else:
            # Standard optimization
            action = self._plan_standard_optimization(state)
            self.last_action = "standard"
            
        self.refinement_history.append({
            "iteration": len(state.configs),
            "action": action["type"],
            "reasoning": action["reasoning"],
            "best_sharpe": state.best_sharpe,
            "convergence": convergence,
        })
        
        return action
    
    def _plan_meta_intervention(self, state: OptimizationState) -> Dict[str, Any]:
        """Plan meta-learning intervention to break stagnation"""
        return {
            "type": "meta_intervention",
            "action": "reset_and_diversify",
            "reasoning": "Stagnation detected, resetting exploration with diverse feature sets",
            "parameters": {
                "reset_bayesian": True,
                "explore_new_features": True,
                "diversify_population": True,
                "increase_exploration": True,
            }
        }
    
    def _plan_exploitation_phase(self, state: OptimizationState) -> Dict[str, Any]:
        """Plan exploitation of best patterns"""
        return {
            "type": "exploitation",
            "action": "refine_best_configs",
            "reasoning": "High convergence detected, focusing on refinement of top performers",
            "parameters": {
                "use_top_configs": 3,
                "small_variations": True,
                "ensemble_exploration": True,
            }
        }
    
    def _plan_exploration_phase(self, state: OptimizationState) -> Dict[str, Any]:
        """Plan broad exploration"""
        return {
            "type": "exploration",
            "action": "diversified_search",
            "reasoning": "Early phase, conducting broad feature space exploration",
            "parameters": {
                "diverse_features": True,
                "wide_parameter_range": True,
                "multiple_approaches": True,
            }
        }
    
    def _plan_standard_optimization(self, state: OptimizationState) -> Dict[str, Any]:
        """Plan standard Bayesian optimization"""
        return {
            "type": "standard",
            "action": "bayesian_optimization",
            "reasoning": "Standard optimization phase with learned preferences",
            "parameters": {
                "use_learned_patterns": True,
                "balance_exploration_exploitation": True,
            }
        }

# Available features for optimization
AVAILABLE_FEATURES = [
    # Market momentum
    "XLY_ma_ratio", "XLB_ma_ratio", "XLI_ma_ratio", "XLF_ma_ratio",
    "XLK_ma_ratio", "XLP_ma_ratio", "XLE_ma_ratio", "XLU_ma_ratio",
    "SPY_ma_ratio", "IWM_ma_ratio",
    # Momentum
    "XLY_mom_63d", "SPY_mom_63d", "IWM_mom_63d",
    # Volatility
    "IEF_vol_21d", "SHY_vol_21d", "TLT_vol_21d", "HYG_vol_21d",
    "vix_zscore", "vix_roc_63d",
    # Rates
    "rate_roc_63d", "yield_curve", "rate_level",
    # Fed policy
    "fed_policy_delta", "fed_sentiment",
    # Macro
    "unrate", "MICH", "UMCSENT",
]


def initialize_smart_optimization(
    factor: str = "VLUE",
    available_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Initialize smart optimization with Bayesian and Genetic strategies.
    Call this at the start of optimization.
    
    Args:
        factor: Factor to optimize
        available_features: List of available features (uses default if not provided)
        
    Returns:
        Initialization status and first suggestions
    """
    global _optimization_state, _bayesian_optimizer, _genetic_optimizer
    
    features = available_features or AVAILABLE_FEATURES
    
    _optimization_state = OptimizationState()
    _bayesian_optimizer = BayesianOptimizer(features)
    _genetic_optimizer = GeneticOptimizer(features, population_size=8)
    _genetic_optimizer.initialize_population()
    
    # Get initial suggestions from Bayesian optimizer
    suggestions = _bayesian_optimizer.suggest_next(3)
    
    return {
        "status": "initialized",
        "factor": factor,
        "n_features_available": len(features),
        "initial_suggestions": suggestions,
        "message": "Smart optimization initialized. Use suggest_next_experiment to get AI-guided suggestions.",
    }


def suggest_next_experiment(factor: str = "VLUE", phase: str = "FEATURE_DISCOVERY") -> Dict[str, Any]:
    """
    Get AI-enhanced suggestions with learning-based recommendations.
    Uses pattern recognition and adaptive exploration.
    
    Args:
        factor: Factor to optimize (VLUE, SIZE, etc.)
        phase: Current optimization phase
    
    Returns:
        Enhanced suggestions with learning insights
    """
    global _optimization_state, _bayesian_optimizer
    
    if _bayesian_optimizer is None:
        return {"error": "Call initialize_smart_optimization first"}
    
    # Get base Bayesian suggestions
    base_suggestions = _bayesian_optimizer.suggest_next(2)
    
    # Enhanced suggestions based on learning
    enhanced_suggestions = []
    
    if _optimization_state and len(_optimization_state.configs) >= 3:
        # Learning-based suggestions
        
        # 1. Exploit successful patterns
        successful_patterns = _optimization_state.memory.successful_patterns
        if successful_patterns:
            pattern_suggestion = _create_pattern_based_suggestion(successful_patterns)
            if pattern_suggestion:
                enhanced_suggestions.append(pattern_suggestion)
        
        # 2. Focus on top-performing features
        if _optimization_state.feature_importance:
            feature_suggestion = _create_feature_focused_suggestion(
                _optimization_state.feature_importance
            )
            if feature_suggestion:
                enhanced_suggestions.append(feature_suggestion)
        
        # 3. Anti-stagnation suggestion if needed
        if _optimization_state.is_stagnating():
            exploration_suggestion = _create_exploration_suggestion()
            if exploration_suggestion:
                enhanced_suggestions.append(exploration_suggestion)
    
    # Combine base and enhanced suggestions
    all_suggestions = base_suggestions + enhanced_suggestions
    
    # Current phase with adaptive logic
    current_phase = _optimization_state.phase if _optimization_state else ExplorationPhase.FEATURE_DISCOVERY
    convergence = _optimization_state.get_convergence_signal() if _optimization_state else 0.0
    
    # Generate adaptive guidance
    guidance_parts = []
    if _optimization_state and _optimization_state.is_stagnating():
        guidance_parts.append("🔄 Breaking stagnation with diverse exploration")
    elif convergence > 0.8:
        guidance_parts.append("🎯 High convergence - focus on refinement")
    elif len(all_suggestions) > 2:
        guidance_parts.append("🧠 Using learned patterns for suggestions")
    else:
        guidance_parts.append(f"📊 Focus on {phase}")
    
    return {
        "suggestions": all_suggestions[:3],  # Limit to 3 best suggestions
        "base_suggestions": base_suggestions,
        "learned_suggestions": enhanced_suggestions,
        "current_phase": current_phase.value,
        "requested_phase": phase,
        "factor": factor,
        "n_tested": len(_optimization_state.configs) if _optimization_state else 0,
        "best_sharpe": _optimization_state.best_sharpe if _optimization_state else 0,
        "convergence": convergence,
        "is_stagnating": _optimization_state.is_stagnating() if _optimization_state else False,
        "learning_insights": {
            "successful_patterns": len(_optimization_state.memory.successful_patterns) if _optimization_state else 0,
            "top_features": list(_optimization_state.feature_importance.keys())[:5] if _optimization_state and _optimization_state.feature_importance else [],
            "stagnation_count": _optimization_state.stagnation_count if _optimization_state else 0,
        },
        "guidance": " | ".join(guidance_parts),
    }


def _create_pattern_based_suggestion(successful_patterns: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create suggestion based on successful patterns"""
    if not successful_patterns:
        return None
    
    # Find the most successful pattern
    best_pattern = None
    best_avg_sharpe = -999
    
    for pattern_name, configs in successful_patterns.items():
        if configs:
            avg_sharpe = sum(c["sharpe"] for c in configs) / len(configs)
            if avg_sharpe > best_avg_sharpe:
                best_avg_sharpe = avg_sharpe
                best_pattern = configs[-1]  # Use most recent successful config
    
    if best_pattern:
        # Create variation of successful pattern
        weights = best_pattern["weights"].copy()
        
        # Add small random variations
        for feature in weights:
            weights[feature] *= random.uniform(0.8, 1.2)
        
        return {
            "weights": weights,
            "threshold": best_pattern["threshold"] * random.uniform(0.9, 1.1),
            "hold_period": max(1, int(best_pattern["hold_period"] * random.uniform(0.8, 1.2))),
            "source": "pattern_exploitation"
        }
    
    return None


def _create_feature_focused_suggestion(feature_importance: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Create suggestion focused on top-performing features"""
    if not feature_importance:
        return None
    
    # Select top 5 features
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    weights = {}
    for feature, importance in top_features:
        # Weight proportional to importance with some randomization
        weight = importance * random.uniform(0.5, 1.5)
        if random.random() < 0.5:  # 50% chance to make negative
            weight *= -1
        weights[feature] = weight
    
    return {
        "weights": weights,
        "threshold": random.uniform(0.01, 0.1),
        "hold_period": random.randint(1, 10),
        "source": "feature_focus"
    }


def _create_exploration_suggestion() -> Dict[str, Any]:
    """Create diverse exploration suggestion to break stagnation"""
    # Select random features from different categories
    feature_categories = {
        "fed": [f for f in AVAILABLE_FEATURES if "fed" in f.lower()],
        "volatility": [f for f in AVAILABLE_FEATURES if "vol" in f.lower()],
        "momentum": [f for f in AVAILABLE_FEATURES if "mom" in f.lower()],
        "rates": [f for f in AVAILABLE_FEATURES if "rate" in f.lower()],
        "market": [f for f in AVAILABLE_FEATURES if f not in [item for sublist in [v for k, v in {"fed": [f for f in AVAILABLE_FEATURES if "fed" in f.lower()], "volatility": [f for f in AVAILABLE_FEATURES if "vol" in f.lower()], "momentum": [f for f in AVAILABLE_FEATURES if "mom" in f.lower()], "rates": [f for f in AVAILABLE_FEATURES if "rate" in f.lower()]}.items()] for item in sublist]],
    }
    
    weights = {}
    # Select 1-2 features from each category
    for category, features in feature_categories.items():
        if features:
            selected = random.sample(features, min(2, len(features)))
            for feature in selected:
                weights[feature] = random.uniform(-1, 1)
    
    return {
        "weights": weights,
        "threshold": random.uniform(0.005, 0.15),
        "hold_period": random.randint(1, 15),
        "source": "exploration"
    }


def record_experiment_result(
    weights: Dict[str, float],
    threshold: float,
    hold_period: int,
    sharpe: float,
    max_drawdown: float = 0.0,
    annual_return: float = 0.0,
    volatility: float = 0.0,
    win_rate: float = 0.0,
    source_method: str = "manual",
) -> Dict[str, Any]:
    """
    Record experiment result with enhanced learning and pattern recognition.
    Call this after each backtest.
    
    Args:
        weights: Feature weights used
        threshold: Threshold used
        hold_period: Hold period used
        sharpe: Resulting Sharpe ratio
        max_drawdown: Maximum drawdown
        annual_return: Annual return
        volatility: Strategy volatility
        win_rate: Win rate of trades
        source_method: Method that generated this config
        
    Returns:
        Enhanced insights with learning patterns
    """
    global _optimization_state, _bayesian_optimizer
    
    if _optimization_state is None:
        _optimization_state = OptimizationState()
    if _bayesian_optimizer is None:
        _bayesian_optimizer = BayesianOptimizer(AVAILABLE_FEATURES)
    
    from datetime import datetime
    
    # Create enhanced config with metadata
    config = Config(
        weights=weights,
        threshold=threshold,
        hold_period=hold_period,
        sharpe=sharpe,
        iteration=len(_optimization_state.configs) + 1,
        max_drawdown=max_drawdown,
        annual_return=annual_return,
        volatility=volatility,
        win_rate=win_rate,
        feature_count=len(weights),
        dominant_features=sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3],
        source_method=source_method,
        timestamp=datetime.now().isoformat(),
    )
    
    # Add to optimization state (triggers learning)
    _optimization_state.add_config(config)
    _bayesian_optimizer.add_observation(config)
    
    # Adaptive phase transitions based on learning
    n_configs = len(_optimization_state.configs)
    convergence = _optimization_state.get_convergence_signal()
    
    # Update phase based on learning signals
    if _optimization_state.is_stagnating():
        # Force exploration if stagnating
        if _optimization_state.phase != ExplorationPhase.FEATURE_DISCOVERY:
            _optimization_state.phase = ExplorationPhase.FEATURE_DISCOVERY
    else:
        # Normal adaptive progression
        if n_configs >= 8 and convergence > 0.6:
            _optimization_state.phase = ExplorationPhase.WEIGHT_OPTIMIZATION
        elif n_configs >= 15 and convergence > 0.7:
            _optimization_state.phase = ExplorationPhase.PARAMETER_TUNING
        elif n_configs >= 25:
            _optimization_state.phase = ExplorationPhase.REFINEMENT
    
    # Generate learning insights
    is_new_best = config.sharpe >= _optimization_state.best_sharpe
    improvement = config.sharpe - _optimization_state.best_sharpe
    stagnation_warning = _optimization_state.is_stagnating()
    
    # Feature learning insights
    feature_insights = []
    if _optimization_state.feature_importance:
        top_feature = max(_optimization_state.feature_importance.items(), key=lambda x: x[1])
        if top_feature[0] in weights:
            feature_insights.append(f"Using top feature {top_feature[0]} (score: {top_feature[1]:.3f})")
    
    # Pattern recognition
    pattern = _optimization_state._analyze_config_pattern(config)
    successful_patterns = len(_optimization_state.memory.successful_patterns)
    
    insight_parts = []
    if is_new_best:
        insight_parts.append(f"🏆 New best! (+{improvement:.3f})")
    else:
        insight_parts.append(f"Best remains {_optimization_state.best_sharpe:.3f}")
    
    if stagnation_warning:
        insight_parts.append(f"⚠️ Stagnation detected, exploring new directions")
    
    insight_parts.extend(feature_insights)
    insight_parts.append(f"Pattern: {pattern} | Convergence: {convergence:.2f}")
    
    return {
        "recorded": True,
        "iteration": config.iteration,
        "sharpe": sharpe,
        "is_new_best": is_new_best,
        "improvement": improvement,
        "best_sharpe": _optimization_state.best_sharpe,
        "current_phase": _optimization_state.phase.value,
        "convergence_signal": convergence,
        "is_stagnating": stagnation_warning,
        "pattern_detected": pattern,
        "successful_patterns_count": successful_patterns,
        "feature_importance_top3": dict(list(_optimization_state.feature_importance.items())[:3]) if _optimization_state.feature_importance else {},
        "insight": " | ".join(insight_parts),
    }


def get_genetic_population() -> Dict[str, Any]:
    """
    Get current genetic algorithm population for testing.
    
    Returns:
        List of configs in the current population
    """
    global _genetic_optimizer
    
    if _genetic_optimizer is None:
        return {"error": "Call initialize_smart_optimization first"}
    
    return {
        "generation": _genetic_optimizer.generation,
        "population_size": len(_genetic_optimizer.population),
        "configs": [c.to_dict() for c in _genetic_optimizer.population],
    }


def evolve_genetic_population(fitness_scores: List[float]) -> Dict[str, Any]:
    """
    Evolve the genetic population based on fitness (Sharpe) scores.
    
    Args:
        fitness_scores: Sharpe ratios for each config in the population
        
    Returns:
        New generation of configs to test
    """
    global _genetic_optimizer
    
    if _genetic_optimizer is None:
        return {"error": "Call initialize_smart_optimization first"}
    
    try:
        new_pop = _genetic_optimizer.evolve(fitness_scores)
        best = _genetic_optimizer.get_best()
        
        return {
            "generation": _genetic_optimizer.generation,
            "new_population": [c.to_dict() for c in new_pop],
            "best_config": best.to_dict() if best else None,
            "best_sharpe": best.sharpe if best else 0,
        }
    except ValueError as e:
        return {"error": str(e)}


def build_ensemble(
    top_n: int = 5,
    method: str = "weighted_average",
) -> Dict[str, Any]:
    """
    Build an ensemble from top configs tested so far.
    
    Args:
        top_n: Number of top configs to include
        method: Combination method (weighted_average, simple_average, best_features)
        
    Returns:
        Ensemble configuration
    """
    global _optimization_state
    
    if _optimization_state is None or not _optimization_state.configs:
        return {"error": "No configs tested yet. Run some experiments first."}
    
    ensemble = EnsembleBuilder.build_ensemble(
        _optimization_state.configs,
        top_n=top_n,
        method=method,
    )
    
    return ensemble


def test_ensemble(
    factor: str = "VLUE",
    top_n: int = 5,
    method: str = "weighted_average",
) -> Dict[str, Any]:
    """
    Build ensemble from top configs and backtest it.
    
    Args:
        factor: Factor to test
        top_n: Number of configs to ensemble
        method: Ensemble method
        
    Returns:
        Ensemble config and backtest results
    """
    global _optimization_state
    
    if _optimization_state is None or not _optimization_state.configs:
        return {"error": "No configs tested yet"}
    
    ensemble = EnsembleBuilder.build_ensemble(
        _optimization_state.configs,
        top_n=top_n,
        method=method,
    )
    
    if not ensemble or "weights" not in ensemble:
        return {"error": "Failed to build ensemble"}
    
    # Backtest the ensemble
    result = run_backtest(
        factor=factor,
        signal_weights=ensemble["weights"],
        threshold=ensemble["threshold"],
        hold_period=ensemble["hold_period"],
    )
    
    return {
        "ensemble": ensemble,
        "backtest": result,
    }


def get_reflection_prompt() -> Dict[str, Any]:
    """
    Get a self-reflection prompt to analyze what's working.
    
    Returns:
        Reflection prompt and analysis
    """
    global _optimization_state
    
    if _optimization_state is None or len(_optimization_state.configs) < 2:
        return {"error": "Need at least 2 experiments for reflection"}
    
    prompt = SelfReflection.generate_reflection_prompt(_optimization_state)
    
    # Generate feature importance analysis
    feature_counts = {}
    feature_sharpes = {}
    
    for config in _optimization_state.configs:
        for feat in config.weights:
            if feat not in feature_counts:
                feature_counts[feat] = 0
                feature_sharpes[feat] = []
            feature_counts[feat] += 1
            feature_sharpes[feat].append(config.sharpe)
    
    # Average Sharpe when feature is used
    feature_impact = {}
    for feat in feature_counts:
        feature_impact[feat] = {
            "times_used": feature_counts[feat],
            "avg_sharpe": round(sum(feature_sharpes[feat]) / len(feature_sharpes[feat]), 3),
        }
    
    # Sort by impact
    sorted_features = sorted(
        feature_impact.items(),
        key=lambda x: x[1]["avg_sharpe"],
        reverse=True
    )
    
    return {
        "reflection_prompt": prompt,
        "feature_analysis": dict(sorted_features[:10]),
        "top_features": [f[0] for f in sorted_features[:5]],
        "bottom_features": [f[0] for f in sorted_features[-3:]],
    }


def validate_out_of_sample(
    factor: str = "VLUE",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate best or specified config on out-of-sample data.
    
    Args:
        factor: Factor to test
        config: Config to validate (uses best if not provided)
        
    Returns:
        Train, validation, and test results
    """
    global _optimization_state
    
    if config is None:
        if _optimization_state is None or _optimization_state.best_config is None:
            return {"error": "No config to validate"}
        config = _optimization_state.best_config.to_dict()
    
    validator = OutOfSampleValidator(train_ratio=0.6, val_ratio=0.2)
    splits = validator.get_date_splits("2016-01-01", "2025-12-31")
    
    results = {}
    
    for split_name, (start, end) in splits.items():
        # Run backtest on each split
        result = run_backtest(
            factor=factor,
            signal_weights=config["weights"],
            threshold=config.get("threshold", 0),
            hold_period=config.get("hold_period", 0),
            start_date=start,
            end_date=end,
        )
        results[split_name] = {
            "period": f"{start} to {end}",
            "sharpe": result["strategy"]["sharpe"] if "strategy" in result else 0,
            "annual_return": result["strategy"].get("annual_return", 0) if "strategy" in result else 0,
        }
    
    # Analyze robustness
    train_sharpe = results["train"]["sharpe"]
    val_sharpe = results["validation"]["sharpe"]
    test_sharpe = results["test"]["sharpe"]
    
    validation = validator.validate_config(
        Config.from_dict(config),
        train_sharpe,
        val_sharpe,
    )
    
    return {
        "config": config,
        "results": results,
        "validation": validation,
        "summary": {
            "train_sharpe": train_sharpe,
            "val_sharpe": val_sharpe,
            "test_sharpe": test_sharpe,
            "is_robust": validation["is_robust"],
            "is_overfit": validation["is_overfit"],
        },
    }


def enable_continuous_refinement(
    factor: str = "VLUE", 
    target_sharpe: float = 1.5, 
    max_iterations: int = 50
) -> Dict[str, Any]:
    """
    Enable autonomous continuous refinement mode.
    The system will continuously optimize without human intervention.
    
    Args:
        factor: Factor to optimize
        target_sharpe: Target Sharpe ratio to achieve
        max_iterations: Maximum iterations before stopping
        
    Returns:
        Continuous refinement status
    """
    global _continuous_mode, _refinement_controller, _optimization_state
    
    if _optimization_state is None:
        return {"error": "Call initialize_smart_optimization first"}
    
    _continuous_mode = True
    _refinement_controller = ContinuousRefinementController(
        target_sharpe=target_sharpe,
        max_stagnation=10
    )
    
    return {
        "enabled": True,
        "target_sharpe": target_sharpe,
        "max_iterations": max_iterations,
        "current_best": _optimization_state.best_sharpe,
        "mode": "autonomous",
        "message": f"Continuous refinement enabled. Target: {target_sharpe}, Current: {_optimization_state.best_sharpe:.3f}"
    }


def execute_autonomous_cycle(factor: str = "VLUE") -> Dict[str, Any]:
    """
    Execute one autonomous optimization cycle.
    This function makes decisions and executes experiments automatically.
    
    Args:
        factor: Factor to optimize
        
    Returns:
        Results of autonomous cycle
    """
    global _continuous_mode, _refinement_controller, _optimization_state
    
    if not _continuous_mode or _refinement_controller is None:
        return {"error": "Enable continuous refinement first"}
    
    if _optimization_state is None:
        return {"error": "No optimization state"}
    
    # Check if we should continue
    if not _refinement_controller.should_continue_refinement(_optimization_state):
        _continuous_mode = False
        return {
            "status": "completed",
            "reason": "Target achieved or maximum stagnation reached",
            "final_sharpe": _optimization_state.best_sharpe,
            "target_sharpe": _refinement_controller.target_sharpe,
            "total_experiments": len(_optimization_state.configs),
        }
    
    # Get autonomous action plan
    action_plan = _refinement_controller.get_next_action(_optimization_state)
    
    # Execute the planned action
    results = _execute_action_plan(factor, action_plan)
    
    return {
        "status": "continuing",
        "action_taken": action_plan,
        "results": results,
        "current_best": _optimization_state.best_sharpe,
        "target_sharpe": _refinement_controller.target_sharpe,
        "iterations_completed": len(_optimization_state.configs),
        "stagnation_count": _optimization_state.stagnation_count,
        "convergence": _optimization_state.get_convergence_signal(),
    }


def _execute_action_plan(factor: str, action_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the autonomous action plan
    """
    global _optimization_state, _bayesian_optimizer, _genetic_optimizer
    
    action_type = action_plan["type"]
    parameters = action_plan.get("parameters", {})
    
    executed_experiments = []
    
    try:
        if action_type == "meta_intervention":
            # Reset and diversify
            if parameters.get("reset_bayesian"):
                _bayesian_optimizer = BayesianOptimizer(AVAILABLE_FEATURES)
            
            # Generate diverse experiments
            experiments = _generate_diverse_experiments(3)
            
        elif action_type == "exploitation":
            # Focus on top configs
            top_configs = _optimization_state.get_top_configs(parameters.get("use_top_configs", 3))
            experiments = _generate_refinement_experiments(top_configs)
            
        elif action_type == "exploration":
            # Broad exploration
            experiments = _generate_exploration_experiments(3)
            
        else:  # standard
            # Standard Bayesian suggestions
            suggestions = suggest_next_experiment(factor)
            experiments = suggestions.get("suggestions", [])[:2]
        
        # Execute each experiment
        for exp in experiments:
            if "weights" in exp:
                # Import run_backtest here to avoid circular imports
                from .backtest import run_backtest
                
                result = run_backtest(
                    factor=factor,
                    signal_weights=exp["weights"],
                    threshold=exp.get("threshold", 0.05),
                    hold_period=exp.get("hold_period", 5)
                )
                
                if "strategy" in result:
                    strategy_result = result["strategy"]
                    record_result = record_experiment_result(
                        weights=exp["weights"],
                        threshold=exp.get("threshold", 0.05),
                        hold_period=exp.get("hold_period", 5),
                        sharpe=strategy_result.get("sharpe", 0),
                        max_drawdown=strategy_result.get("max_drawdown", 0),
                        annual_return=strategy_result.get("annual_return", 0),
                        volatility=strategy_result.get("volatility", 0),
                        source_method="autonomous"
                    )
                    
                    executed_experiments.append({
                        "config": exp,
                        "sharpe": strategy_result.get("sharpe", 0),
                        "recorded": record_result
                    })
        
        return {
            "executed_count": len(executed_experiments),
            "experiments": executed_experiments,
            "action_type": action_type,
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "action_type": action_type,
            "success": False
        }


def _generate_diverse_experiments(count: int) -> List[Dict[str, Any]]:
    """Generate diverse experimental configurations"""
    experiments = []
    
    # Different feature categories
    categories = {
        "fed_focused": [f for f in AVAILABLE_FEATURES if "fed" in f.lower() or "policy" in f.lower()],
        "volatility_focused": [f for f in AVAILABLE_FEATURES if "vol" in f.lower() or "vix" in f.lower()],
        "momentum_focused": [f for f in AVAILABLE_FEATURES if "mom" in f.lower() or "roc" in f.lower()],
    }
    
    for i, (category, features) in enumerate(categories.items()):
        if i >= count:
            break
        if features:
            weights = {}
            selected_features = random.sample(features, min(4, len(features)))
            for feature in selected_features:
                weights[feature] = random.uniform(-1, 1)
            
            experiments.append({
                "weights": weights,
                "threshold": random.uniform(0.01, 0.1),
                "hold_period": random.randint(1, 10),
                "source": f"diverse_{category}"
            })
    
    return experiments


def _generate_refinement_experiments(top_configs: List[Config]) -> List[Dict[str, Any]]:
    """Generate refinement experiments based on top configs"""
    experiments = []
    
    for config in top_configs[:2]:
        # Small variations of successful configs
        weights = config.weights.copy()
        for feature in weights:
            weights[feature] *= random.uniform(0.9, 1.1)
        
        experiments.append({
            "weights": weights,
            "threshold": config.threshold * random.uniform(0.9, 1.1),
            "hold_period": max(1, int(config.hold_period * random.uniform(0.9, 1.1))),
            "source": "refinement"
        })
    
    return experiments


def _generate_exploration_experiments(count: int) -> List[Dict[str, Any]]:
    """Generate broad exploration experiments"""
    experiments = []
    
    for _ in range(count):
        # Random feature selection
        n_features = random.randint(2, 6)
        selected_features = random.sample(AVAILABLE_FEATURES, n_features)
        
        weights = {}
        for feature in selected_features:
            weights[feature] = random.uniform(-1.5, 1.5)
        
        experiments.append({
            "weights": weights,
            "threshold": random.uniform(0.005, 0.15),
            "hold_period": random.randint(1, 15),
            "source": "exploration"
        })
    
    return experiments


def get_optimization_summary() -> Dict[str, Any]:
    """
    Get full summary of optimization with continuous refinement status.
    
    Returns:
        Enhanced summary with autonomous optimization insights
    """
    global _optimization_state, _continuous_mode, _refinement_controller
    
    if _optimization_state is None or not _optimization_state.configs:
        return {"error": "No optimization data yet"}
    
    top_configs = _optimization_state.get_top_configs(5)
    
    base_summary = {
        "n_experiments": len(_optimization_state.configs),
        "best_sharpe": _optimization_state.best_sharpe,
        "best_config": _optimization_state.best_config.to_dict() if _optimization_state.best_config else None,
        "current_phase": _optimization_state.phase.value,
        "top_5_configs": [c.to_dict() for c in top_configs],
        "sharpe_progression": [c.sharpe for c in _optimization_state.configs],
        "convergence_signal": _optimization_state.get_convergence_signal(),
        "stagnation_count": _optimization_state.stagnation_count,
    }
    
    # Add continuous refinement status
    if _continuous_mode and _refinement_controller:
        base_summary.update({
            "continuous_mode": True,
            "target_sharpe": _refinement_controller.target_sharpe,
            "progress_to_target": min(1.0, _optimization_state.best_sharpe / _refinement_controller.target_sharpe),
            "refinement_history": _refinement_controller.refinement_history[-10:],  # Last 10 actions
            "last_action": _refinement_controller.last_action,
            "should_continue": _refinement_controller.should_continue_refinement(_optimization_state),
        })
    else:
        base_summary["continuous_mode"] = False
    
    return base_summary


def deploy_best_strategy(
    factor: str = "VLUE",
    minimum_sharpe: float = 0.5,
    confidence_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Deploy the best strategy configuration when improvement criteria are met.
    
    Args:
        factor: Factor to deploy strategy for
        minimum_sharpe: Minimum Sharpe required for deployment
        confidence_threshold: Minimum confidence score for deployment
        
    Returns:
        Deployment status and configuration
    """
    global _optimization_state
    
    if _optimization_state is None or not _optimization_state.configs:
        return {"error": "No optimization data for deployment"}
    
    best_config = _optimization_state.best_config
    if not best_config:
        return {"error": "No best configuration found"}
    
    # Check deployment criteria
    meets_sharpe = best_config.sharpe >= minimum_sharpe
    convergence = _optimization_state.get_convergence_signal()
    meets_confidence = convergence >= confidence_threshold
    
    # Additional validation - out of sample performance
    oos_validation = validate_out_of_sample(factor, best_config.to_dict())
    is_robust = oos_validation.get("summary", {}).get("is_robust", False)
    
    deployment_score = _calculate_deployment_score(
        best_config, convergence, oos_validation
    )
    
    can_deploy = meets_sharpe and meets_confidence and is_robust
    
    if can_deploy:
        # Save strategy configuration
        strategy_config = _create_strategy_config(factor, best_config)
        deployment_result = _save_strategy_deployment(factor, strategy_config)
        
        return {
            "deployed": True,
            "factor": factor,
            "strategy_config": strategy_config,
            "deployment_score": deployment_score,
            "performance_metrics": {
                "sharpe": best_config.sharpe,
                "max_drawdown": best_config.max_drawdown,
                "annual_return": best_config.annual_return,
                "convergence": convergence,
            },
            "validation": oos_validation.get("summary", {}),
            "deployment_file": deployment_result.get("file_path"),
            "message": f"🚀 Strategy deployed for {factor} (Sharpe: {best_config.sharpe:.3f})",
        }
    else:
        return {
            "deployed": False,
            "factor": factor,
            "reason": _get_deployment_rejection_reason(
                meets_sharpe, meets_confidence, is_robust
            ),
            "current_sharpe": best_config.sharpe,
            "required_sharpe": minimum_sharpe,
            "confidence": convergence,
            "required_confidence": confidence_threshold,
            "is_robust": is_robust,
            "deployment_score": deployment_score,
            "message": "Strategy not ready for deployment - needs improvement",
        }


def auto_deploy_on_improvement(
    factor: str = "VLUE",
    auto_deploy: bool = True,
    minimum_improvement: float = 0.05,
) -> Dict[str, Any]:
    """
    Automatically deploy strategy when significant improvement is detected.
    
    Args:
        factor: Factor to monitor
        auto_deploy: Whether to actually deploy or just recommend
        minimum_improvement: Minimum Sharpe improvement to trigger deployment
        
    Returns:
        Auto-deployment status
    """
    global _optimization_state
    
    if _optimization_state is None:
        return {"error": "No optimization state"}
    
    # Check if there's been significant improvement
    recent_configs = _optimization_state.configs[-5:] if len(_optimization_state.configs) >= 5 else _optimization_state.configs
    
    if len(recent_configs) < 2:
        return {"status": "waiting", "message": "Need more experiments for auto-deployment"}
    
    current_best = _optimization_state.best_sharpe
    previous_best = max([c.sharpe for c in _optimization_state.configs[:-1]]) if len(_optimization_state.configs) > 1 else 0
    
    improvement = current_best - previous_best
    significant_improvement = improvement >= minimum_improvement
    
    # Check if strategy meets deployment criteria
    deployment_check = deploy_best_strategy(factor, minimum_sharpe=0.3, confidence_threshold=0.7)
    
    if significant_improvement and auto_deploy and deployment_check.get("deployed"):
        return {
            "auto_deployed": True,
            "improvement": improvement,
            "new_sharpe": current_best,
            "previous_best": previous_best,
            "deployment_info": deployment_check,
            "message": f"🎯 Auto-deployed improved strategy! +{improvement:.3f} Sharpe",
        }
    elif significant_improvement and not auto_deploy:
        return {
            "auto_deployed": False,
            "recommendation": "deploy",
            "improvement": improvement,
            "new_sharpe": current_best,
            "deployment_check": deployment_check,
            "message": f"📈 Significant improvement detected (+{improvement:.3f}). Consider deployment.",
        }
    else:
        return {
            "auto_deployed": False,
            "status": "monitoring",
            "improvement": improvement,
            "current_best": current_best,
            "threshold": minimum_improvement,
            "message": f"Monitoring for improvements (current: {current_best:.3f}, threshold: +{minimum_improvement})",
        }


def _calculate_deployment_score(
    config: Config, convergence: float, oos_validation: Dict[str, Any]
) -> float:
    """
    Calculate a deployment confidence score (0-1)
    """
    # Performance score (0-1)
    sharpe_score = min(1.0, max(0, config.sharpe / 1.5))  # 1.5 Sharpe = perfect score
    
    # Stability score based on convergence
    stability_score = convergence
    
    # Robustness score from out-of-sample validation
    oos_summary = oos_validation.get("summary", {})
    if oos_summary.get("is_robust", False):
        robustness_score = min(1.0, oos_summary.get("test_sharpe", 0) / max(0.01, oos_summary.get("train_sharpe", 0.01)))
    else:
        robustness_score = 0.3  # Penalty for non-robust strategy
    
    # Weighted average
    deployment_score = (
        0.4 * sharpe_score +
        0.3 * stability_score +
        0.3 * robustness_score
    )
    
    return round(deployment_score, 3)


def _get_deployment_rejection_reason(
    meets_sharpe: bool, meets_confidence: bool, is_robust: bool
) -> str:
    """
    Generate human-readable deployment rejection reason
    """
    reasons = []
    if not meets_sharpe:
        reasons.append("Sharpe ratio too low")
    if not meets_confidence:
        reasons.append("Low convergence confidence")
    if not is_robust:
        reasons.append("Failed out-of-sample validation")
    
    return " | ".join(reasons)


def _create_strategy_config(factor: str, config: Config) -> Dict[str, Any]:
    """
    Create deployable strategy configuration
    """
    from datetime import datetime
    
    return {
        "strategy_id": f"{factor}_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "factor": factor,
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "performance": {
            "sharpe": config.sharpe,
            "annual_return": config.annual_return,
            "max_drawdown": config.max_drawdown,
            "volatility": config.volatility,
            "win_rate": config.win_rate,
        },
        "parameters": {
            "signal_weights": config.weights,
            "threshold": config.threshold,
            "hold_period": config.hold_period,
        },
        "metadata": {
            "optimization_method": config.source_method,
            "feature_count": config.feature_count,
            "dominant_features": config.dominant_features,
            "optimization_iteration": config.iteration,
        },
        "deployment_criteria": {
            "minimum_sharpe": 0.5,
            "confidence_threshold": 0.8,
            "out_of_sample_validated": True,
        }
    }


def _save_strategy_deployment(factor: str, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save strategy configuration to deployment file
    """
    import json
    from pathlib import Path
    
    # Create deployments directory
    deployment_dir = Path("deployments")
    deployment_dir.mkdir(exist_ok=True)
    
    # Save strategy config
    strategy_file = deployment_dir / f"{strategy_config['strategy_id']}.json"
    with open(strategy_file, "w") as f:
        json.dump(strategy_config, f, indent=2)
    
    # Update current strategy symlink
    current_strategy_link = deployment_dir / f"current_{factor}.json"
    if current_strategy_link.exists():
        current_strategy_link.unlink()
    
    # Create symlink to current strategy (cross-platform compatible)
    try:
        current_strategy_link.symlink_to(strategy_file.name)
    except OSError:
        # Fallback for systems without symlink support
        with open(current_strategy_link, "w") as f:
            json.dump(strategy_config, f, indent=2)
    
    return {
        "file_path": str(strategy_file),
        "current_link": str(current_strategy_link),
        "strategy_id": strategy_config["strategy_id"],
    }


def get_deployed_strategy(factor: str = "VLUE") -> Dict[str, Any]:
    """
    Get the currently deployed strategy for a factor.
    
    Args:
        factor: Factor to get deployed strategy for
        
    Returns:
        Current deployed strategy configuration
    """
    import json
    from pathlib import Path
    
    deployment_dir = Path("deployments")
    current_strategy_file = deployment_dir / f"current_{factor}.json"
    
    if not current_strategy_file.exists():
        return {
            "deployed": False,
            "message": f"No deployed strategy found for {factor}",
        }
    
    try:
        with open(current_strategy_file, "r") as f:
            strategy_config = json.load(f)
        
        return {
            "deployed": True,
            "factor": factor,
            "strategy_config": strategy_config,
            "file_path": str(current_strategy_file),
        }
    
    except Exception as e:
        return {
            "error": f"Failed to load deployed strategy: {e}",
            "deployed": False,
        }


def detect_overfitting_signals(
    config: Dict[str, Any],
    backtest_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detect overfitting signals focusing on spurious correlations without fundamental basis.
    
    Args:
        config: Strategy configuration to check
        backtest_result: Optional backtest results to validate
        
    Returns:
        Overfitting analysis with warnings focusing on unexplainable correlations
    """
    warnings = []
    severity_scores = []
    
    # Check configuration for overfitting signals
    weights = config.get("weights", {})
    
    # 1. Feature count complexity
    feature_count = len(weights)
    if feature_count > 8:
        warnings.append(f"High feature count ({feature_count}) - risk of spurious correlations")
        severity_scores.append(0.6)
    
    # 2. Analyze features for fundamental vs spurious correlations
    spurious_features = _identify_spurious_correlations(weights)
    
    for feature_info in spurious_features:
        feature = feature_info["feature"]
        reason = feature_info["reason"]
        ic = feature_info.get("ic", 0)
        
        if feature_info["severity"] == "high":
            warnings.append(f"SPURIOUS: {feature} - {reason} (IC: {ic:.3f})")
            severity_scores.append(0.9)
        elif feature_info["severity"] == "medium":
            warnings.append(f"QUESTIONABLE: {feature} - {reason} (IC: {ic:.3f})")
            severity_scores.append(0.6)
        else:
            warnings.append(f"MONITOR: {feature} - {reason} (IC: {ic:.3f})")
            severity_scores.append(0.4)
    
    # 3. Check for features that are too perfectly correlated
    perfect_correlations = _check_perfect_correlations(weights)
    for feature, ic in perfect_correlations:
        warnings.append(f"PERFECT CORRELATION: {feature} (IC: {ic:.3f}) - likely data mining artifact")
        severity_scores.append(0.8)
    
    # 3. Check backtest results if provided
    if backtest_result and "strategy" in backtest_result:
        strategy_stats = backtest_result["strategy"]
        
        sharpe = strategy_stats.get("sharpe", 0)
        max_dd = abs(strategy_stats.get("max_drawdown", 0))
        annual_return = strategy_stats.get("annual_return", 0)
        hit_rate = strategy_stats.get("hit_rate", 0.5)
        
        # Overfitting red flags in results
        if sharpe > 3.0:
            warnings.append(f"Extremely high Sharpe ({sharpe:.2f}) - likely overfit")
            severity_scores.append(0.9)
        elif sharpe > 2.0:
            warnings.append(f"Very high Sharpe ({sharpe:.2f}) - validate carefully")
            severity_scores.append(0.6)
        
        if max_dd < 1.0 and sharpe > 1.0:
            warnings.append(f"Unrealistically low max DD ({max_dd:.1f}%) for Sharpe {sharpe:.2f}")
            severity_scores.append(0.8)
        
        if annual_return > 0.5:  # 50%+ annual return
            warnings.append(f"Unsustainable annual return ({annual_return*100:.1f}%)")
            severity_scores.append(0.7)
        
        if hit_rate > 0.85:
            warnings.append(f"Suspiciously high hit rate ({hit_rate*100:.1f}%)")
            severity_scores.append(0.6)
        
        # Check for perfect performance
        if sharpe > 2.5 and max_dd < 2.0 and hit_rate > 0.8:
            warnings.append("Perfect trifecta (high Sharpe + low DD + high hit rate) - classic overfit")
            severity_scores.append(1.0)
    
    # Calculate overall concern level (informational, not prescriptive)
    concern_level = max(severity_scores) if severity_scores else 0.0
    
    # Provide guidance, not hard recommendations
    if concern_level >= 0.8:
        guidance = "High concern signals detected - carefully evaluate economic rationale"
    elif concern_level >= 0.6:
        guidance = "Moderate concerns - consider out-of-sample validation"
    elif concern_level >= 0.4:
        guidance = "Some potential issues - worth investigating"
    else:
        guidance = "No major red flags detected"
    
    return {
        "concern_level": concern_level,
        "warnings": warnings,
        "guidance": guidance,
        "feature_count": feature_count,
        "analysis_note": "This is informational analysis - use your judgment to make decisions",
        "key_questions": [
            "Do these features have clear economic mechanisms?",
            "Are the correlations explainable or suspicious?",
            "Will these relationships likely persist out-of-sample?"
        ]
    }


def apply_overfitting_filters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide information for intelligent overfitting assessment, but let the model make decisions.
    This is informational only - the model should use judgment rather than hard rules.
    
    Args:
        config: Original configuration
        
    Returns:
        Configuration with analysis metadata for model judgment
    """
    analysis_config = config.copy()
    weights = config.get("weights", {})
    
    # Provide analysis for model judgment
    feature_analysis = []
    for feature, weight in weights.items():
        reasoning = _get_fundamental_reasoning(feature)
        
        feature_analysis.append({
            "feature": feature,
            "weight": weight,
            "economic_reasoning": reasoning["reasoning"],
            "mechanism": reasoning["mechanism"], 
            "fundamental_strength": reasoning["strength"],
            "judgment_notes": f"Weight: {weight:.3f}, Strength: {reasoning['strength']}"
        })
    
    # Add metadata for model consideration (not decisions)
    analysis_config["_overfitting_analysis"] = {
        "total_features": len(weights),
        "feature_analysis": feature_analysis,
        "guidance": "Use this analysis to make informed judgments about feature validity",
        "note": "No automatic filtering applied - use your economic reasoning"
    }
    
    return analysis_config


def validate_robustness(
    factor: str, 
    config: Dict[str, Any], 
    min_train_val_correlation: float = 0.5
) -> Dict[str, Any]:
    """
    Validate strategy robustness across different time periods.
    
    Args:
        factor: Factor to test
        config: Configuration to validate
        min_train_val_correlation: Minimum correlation between train/val performance
        
    Returns:
        Robustness validation results
    """
    # Run out-of-sample validation
    oos_results = validate_out_of_sample(factor, config)
    
    if "results" not in oos_results:
        return {"error": "Failed to run out-of-sample validation"}
    
    results = oos_results["results"]
    train_sharpe = results.get("train", {}).get("sharpe", 0)
    val_sharpe = results.get("validation", {}).get("sharpe", 0)
    test_sharpe = results.get("test", {}).get("sharpe", 0)
    
    # Check for overfitting patterns
    train_val_correlation = min(1.0, val_sharpe / max(0.01, train_sharpe))
    val_test_correlation = min(1.0, test_sharpe / max(0.01, val_sharpe))
    
    is_overfit = (
        train_val_correlation < min_train_val_correlation or
        val_test_correlation < 0.3 or  # Severe degradation in test
        train_sharpe > 2.0 and test_sharpe < 0.5  # Classic overfit pattern
    )
    
    degradation_train_to_test = (train_sharpe - test_sharpe) / max(0.01, train_sharpe)
    
    return {
        "is_overfit": is_overfit,
        "train_sharpe": train_sharpe,
        "val_sharpe": val_sharpe,
        "test_sharpe": test_sharpe,
        "train_val_correlation": train_val_correlation,
        "val_test_correlation": val_test_correlation,
        "degradation_pct": degradation_train_to_test * 100,
        "robustness_score": min(train_val_correlation, val_test_correlation),
        "recommendation": (
            "REJECT - Overfitting detected" if is_overfit else
            "ACCEPT - Robust across periods" if min(train_val_correlation, val_test_correlation) > 0.7 else
            "CAUTION - Moderate robustness concerns"
        )
    }


def get_overfitting_summary() -> Dict[str, Any]:
    """
    Get summary of overfitting patterns detected in current optimization.
    
    Returns:
        Summary of overfitting incidents and patterns
    """
    global _optimization_state
    
    if _optimization_state is None or not _optimization_state.configs:
        return {"error": "No optimization data"}
    
    overfit_configs = []
    suspicious_configs = []
    
    for config in _optimization_state.configs:
        # Analyze each config for overfitting
        analysis = detect_overfitting_signals(config.to_dict())
        
        if analysis["should_reject"]:
            overfit_configs.append({
                "iteration": config.iteration,
                "sharpe": config.sharpe,
                "risk_score": analysis["risk_score"],
                "warnings": analysis["warnings"]
            })
        elif analysis["needs_validation"]:
            suspicious_configs.append({
                "iteration": config.iteration,
                "sharpe": config.sharpe,
                "risk_score": analysis["risk_score"],
                "warnings": analysis["warnings"]
            })
    
    # Analyze patterns
    high_sharpe_configs = [c for c in _optimization_state.configs if c.sharpe > 2.0]
    complex_configs = [c for c in _optimization_state.configs if len(c.weights) > 6]
    
    return {
        "total_configs": len(_optimization_state.configs),
        "overfit_configs": len(overfit_configs),
        "suspicious_configs": len(suspicious_configs),
        "overfit_rate": len(overfit_configs) / len(_optimization_state.configs),
        "high_sharpe_count": len(high_sharpe_configs),
        "complex_configs": len(complex_configs),
        "overfit_details": overfit_configs,
        "suspicious_details": suspicious_configs,
        "patterns": {
            "avg_features_in_overfit": sum(len(c.weights) for c in _optimization_state.configs if len(c.weights) > 6) / max(1, len(complex_configs)),
            "max_sharpe_detected": max((c.sharpe for c in _optimization_state.configs), default=0),
        },
        "recommendations": [
            "Use feature count <= 6 to reduce overfitting",
            "Validate all Sharpe > 2.0 with out-of-sample testing",
            "Monitor for degradation in test vs train performance",
            "Apply overfitting filters before testing configs"
        ]
    }


def _identify_spurious_correlations(weights: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Identify features that may have spurious correlations without fundamental basis.
    
    Args:
        weights: Feature weights to analyze
        
    Returns:
        List of suspicious features with reasoning
    """
    spurious_features = []
    
    # Define fundamental feature categories with economic rationale
    FUNDAMENTAL_CATEGORIES = {
        # Strong fundamental basis
        "fed_policy": ["fed_policy_stance", "fed_sentiment", "fed_policy_delta", "MICH", "UMCSENT"],
        "macro_conditions": ["unrate", "yield_curve", "rate_level", "rate_roc_63d"],
        "market_stress": ["vix_zscore", "vix_roc_63d", "IEF_vol_21d", "TLT_vol_21d", "HYG_vol_21d"],
        "sector_momentum": ["XLY_mom_63d", "SPY_mom_63d", "IWM_mom_63d"],
        
        # Medium fundamental basis (require more scrutiny)
        "technical_indicators": ["SPY_ma_ratio", "IWM_ma_ratio", "XLY_ma_ratio", "XLB_ma_ratio"],
        "bond_signals": ["IEF_vol_21d", "SHY_vol_21d", "TLT_vol_21d"],
        
        # Weak fundamental basis (high spurious risk)
        "sector_ratios": ["XLF_ma_ratio", "XLK_ma_ratio", "XLP_ma_ratio", "XLE_ma_ratio", "XLU_ma_ratio"],
        "obscure_volatility": ["SHY_vol_21d"],  # Short-term treasury vol is often noise
    }
    
    # Reverse mapping: feature -> category
    feature_to_category = {}
    for category, features in FUNDAMENTAL_CATEGORIES.items():
        for feature in features:
            feature_to_category[feature] = category
    
    for feature, weight in weights.items():
        category = feature_to_category.get(feature, "unknown")
        
        # Provide observations for model judgment (not automatic rejections)
        if category == "unknown":
            spurious_features.append({
                "feature": feature,
                "reason": "Consider: What economic mechanism connects this feature to factor performance?",
                "concern_level": "investigate",
                "category": "unknown"
            })
        
        elif category == "sector_ratios" and abs(weight) > 0.5:
            spurious_features.append({
                "feature": feature,
                "reason": "Question: Why would sector technical levels predict broader factor timing?",
                "concern_level": "moderate",
                "category": category
            })
        
        elif category == "obscure_volatility":
            spurious_features.append({
                "feature": feature,
                "reason": "Consider: Is short-term bond volatility meaningful for factor prediction?",
                "concern_level": "moderate", 
                "category": category
            })
        
        elif category == "technical_indicators" and abs(weight) > 1.0:
            spurious_features.append({
                "feature": feature,
                "reason": "High weight on technical indicator - evaluate economic justification",
                "concern_level": "review",
                "category": category
            })
        
        # Flag potential regime-specific patterns
        elif feature.endswith("_ma_ratio") and abs(weight) > 0.8:
            spurious_features.append({
                "feature": feature,
                "reason": "High weight on moving average - consider if this reflects regime-specific patterns",
                "concern_level": "review",
                "category": "technical_heavy"
            })
    
    return spurious_features


def _check_perfect_correlations(weights: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    Check for features with suspiciously perfect correlations.
    
    Args:
        weights: Feature weights to check
        
    Returns:
        List of (feature, ic) tuples for features with perfect correlations
    """
    perfect_correlations = []
    
    try:
        # Import feature correlation function
        from .analysis import get_factor_data
        
        # Get feature correlations for the most recent period
        features_df, _, _ = get_factor_data("VLUE", "2020-01-01")  # Recent period for correlation check
        
        for feature in weights.keys():
            if feature in features_df.columns:
                # Calculate correlation with VLUE returns (as proxy for IC)
                vlue_returns = features_df["VLUE_fwd_21d_return"].dropna()
                feature_values = features_df[feature].dropna()
                
                # Align indices
                common_idx = vlue_returns.index.intersection(feature_values.index)
                if len(common_idx) > 50:  # Enough data points
                    corr = vlue_returns.loc[common_idx].corr(feature_values.loc[common_idx])
                    
                    # Flag suspiciously high correlations
                    if abs(corr) > 0.85:
                        perfect_correlations.append((feature, corr))
    
    except Exception:
        # If we can't calculate correlations, flag based on heuristics
        for feature, weight in weights.items():
            if abs(weight) > 1.5:  # Suspiciously high weight
                perfect_correlations.append((feature, weight))
    
    return perfect_correlations


def _get_fundamental_reasoning(feature: str) -> Dict[str, Any]:
    """
    Get fundamental economic reasoning for why a feature should predict factor performance.
    
    Args:
        feature: Feature name to analyze
        
    Returns:
        Fundamental reasoning and strength assessment
    """
    # Economic reasoning for major feature categories
    FUNDAMENTAL_REASONING = {
        # Strong fundamental basis
        "fed_policy_stance": {
            "reasoning": "Fed policy directly affects discount rates and risk premiums, driving style factor performance",
            "strength": "strong",
            "mechanism": "Policy tightening increases discount rates, hurting growth vs value"
        },
        "fed_sentiment": {
            "reasoning": "Fed communication affects market expectations and risk appetite", 
            "strength": "strong",
            "mechanism": "Hawkish sentiment increases rates expectation, benefits value over growth"
        },
        "unrate": {
            "reasoning": "Employment affects economic cycle and factor performance",
            "strength": "strong", 
            "mechanism": "Low unemployment signals economic strength, benefits cyclical factors"
        },
        "vix_zscore": {
            "reasoning": "Market stress affects factor style performance through risk appetite",
            "strength": "strong",
            "mechanism": "High VIX benefits quality/low vol, hurts momentum/small cap"
        },
        "yield_curve": {
            "reasoning": "Yield curve shape reflects growth expectations affecting factor performance",
            "strength": "strong",
            "mechanism": "Steep curve suggests growth expectations, benefits small cap/momentum"
        },
        
        # Medium fundamental basis
        "SPY_mom_63d": {
            "reasoning": "Market momentum affects factor rotation through sentiment",
            "strength": "medium",
            "mechanism": "Strong market momentum benefits momentum factor, hurts quality"
        },
        "XLY_mom_63d": {
            "reasoning": "Consumer discretionary momentum reflects economic confidence",
            "strength": "medium", 
            "mechanism": "XLY strength suggests economic growth, benefits cyclical factors"
        },
        
        # Weak/questionable basis
        "XLE_ma_ratio": {
            "reasoning": "Energy sector technical signals have weak factor predictive power",
            "strength": "weak",
            "mechanism": "Unclear why energy moving averages should predict non-energy factor performance"
        },
        "XLU_ma_ratio": {
            "reasoning": "Utility technical signals often reflect interest rate noise",
            "strength": "weak",
            "mechanism": "Utilities are rate-sensitive but connection to factor timing unclear"
        }
    }
    
    if feature in FUNDAMENTAL_REASONING:
        return FUNDAMENTAL_REASONING[feature]
    
    # Heuristic reasoning for unknown features
    if "fed" in feature.lower() or "policy" in feature.lower():
        return {
            "reasoning": "Fed-related features have strong fundamental basis for factor prediction",
            "strength": "strong",
            "mechanism": "Monetary policy affects discount rates and style factor relative performance"
        }
    elif "vix" in feature.lower() or "vol" in feature.lower():
        return {
            "reasoning": "Volatility measures reflect risk appetite affecting factor styles",
            "strength": "medium", 
            "mechanism": "High volatility environments favor defensive factors over aggressive ones"
        }
    elif "mom" in feature.lower():
        return {
            "reasoning": "Momentum measures reflect market sentiment trends",
            "strength": "medium",
            "mechanism": "Broad momentum affects factor style rotation through risk appetite"
        }
    elif "_ma_ratio" in feature:
        return {
            "reasoning": "Moving average ratios are often technical noise without fundamental basis",
            "strength": "weak", 
            "mechanism": "Unclear economic mechanism connecting technical levels to fundamental factor drivers"
        }
    elif "rate" in feature.lower():
        return {
            "reasoning": "Interest rates affect discount rates and factor valuations",
            "strength": "strong",
            "mechanism": "Rate changes affect relative valuation of growth vs value styles"
        }
    else:
        return {
            "reasoning": "No clear fundamental economic rationale identified",
            "strength": "unknown",
            "mechanism": "Unclear connection to fundamental factor drivers"
        }



def diagnose_underperformance(
    factor: str,
    signal_weights: Dict[str, float],
    threshold: float = 0.03,
    target_type: str = "ratio",
    benchmark: str = "SPY",
) -> Dict[str, Any]:
    """
    Analyze periods where the strategy underperformed and diagnose WHY.
    
    Returns detailed analysis of drawdown periods including:
    - What position the strategy took vs what was correct
    - Feature values during those periods
    - Suggested improvements
    
    Args:
        factor: Factor ETF symbol
        signal_weights: Current signal weights to diagnose
        threshold: Signal threshold
        target_type: "premium" or "ratio"
        benchmark: Benchmark symbol
        
    Returns:
        Dict with diagnostic information for LLM to analyze
    """
    import pandas as pd
    import numpy as np
    from .analysis import get_factor_data
    from .backtest import _compute_weighted_signal
    
    # Get data
    features, factor_ret, premium = get_factor_data(factor, "2016-01-01", benchmark=benchmark)
    
    # Compute signal and positions
    signal = _compute_weighted_signal(features, signal_weights)
    positions = pd.Series(0.0, index=signal.index)
    positions[signal > threshold] = 1.0
    positions[signal < -threshold] = -1.0
    positions = positions.shift(1).fillna(0)
    
    # OOS period
    oos_start = 504
    positions_oos = positions.iloc[oos_start:]
    premium_oos = premium.iloc[oos_start:]
    
    # Strategy returns and drawdowns
    strategy_ret = positions_oos * premium_oos
    strategy_ret = strategy_ret.dropna()
    equity = (1 + strategy_ret).cumprod()
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    
    # Find significant drawdown periods (DD < -5%)
    dd_periods = []
    in_dd = False
    start = None
    
    for date, dd in drawdown.items():
        if dd < -0.05 and not in_dd:
            in_dd = True
            start = date
        elif dd >= -0.02 and in_dd:
            in_dd = False
            dd_periods.append((start, date))
            start = None
    
    if in_dd and start:
        dd_periods.append((start, drawdown.index[-1]))
    
    # Analyze each period
    diagnostics = []
    key_features = ['yield_curve', 'XLY_ma_ratio', 'XLK_mom_63d', 'IWM_ma_ratio', 
                    'rate_roc_63d', 'TLT_mom_63d', 'VIX', 'unrate', 'speech_policy_stance']
    
    for start, end in dd_periods[:5]:  # Top 5 periods
        period_dd = drawdown.loc[start:end].min()
        period_premium = premium_oos.loc[start:end]
        period_pos = positions_oos.loc[start:end]
        period_features = features.loc[start:end]
        
        # What actually happened
        actual_move = period_premium.sum()
        actual_direction = "UP" if actual_move > 0 else "DOWN"
        our_position = "LONG" if period_pos.mode().iloc[0] > 0 else "SHORT" if period_pos.mode().iloc[0] < 0 else "NEUTRAL"
        was_correct = (our_position == "LONG" and actual_move > 0) or \
                      (our_position == "SHORT" and actual_move < 0) or \
                      our_position == "NEUTRAL"
        
        # Feature values
        feature_values = {}
        for feat in key_features:
            if feat in period_features.columns:
                feature_values[feat] = round(period_features[feat].mean(), 3)
        
        diagnostics.append({
            "period": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
            "drawdown_pct": round(period_dd * 100, 1),
            "factor_vs_benchmark_move": f"{actual_direction} ({actual_move*100:.1f}%)",
            "our_position": our_position,
            "was_correct": was_correct,
            "feature_values": feature_values,
        })
    
    # Summary
    wrong_calls = [d for d in diagnostics if not d["was_correct"]]
    
    return {
        "factor": factor,
        "target_type": target_type,
        "total_drawdown_periods": len(dd_periods),
        "analyzed_periods": len(diagnostics),
        "wrong_calls": len(wrong_calls),
        "diagnostics": diagnostics,
        "available_features": key_features,
        "suggestion": "Look for features that could have predicted the 'wrong call' periods. "
                      "Consider adding inverse signals for growth/tech momentum when value underperforms."
    }
