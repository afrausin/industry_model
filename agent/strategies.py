"""
Advanced Optimization Strategies
================================

Implements smarter optimization approaches:
1. Bayesian Optimization - Model the Sharpe surface, pick optimal experiments
2. Genetic Algorithms - Evolve configs through mutation and crossover
3. Structured Exploration - Phase-based search strategy
4. Ensemble Methods - Combine top configs for robustness
5. Self-Reflection - Analyze why configs work/fail
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class ExplorationPhase(Enum):
    """Structured exploration phases"""
    FEATURE_DISCOVERY = "feature_discovery"      # Find best features
    WEIGHT_OPTIMIZATION = "weight_optimization"  # Optimize weights
    PARAMETER_TUNING = "parameter_tuning"        # Tune threshold/hold
    REFINEMENT = "refinement"                    # Fine-tune best config


@dataclass
class Config:
    """Represents a tested configuration with enhanced tracking"""
    weights: Dict[str, float]
    threshold: float = 0.0
    hold_period: int = 0
    sharpe: float = 0.0
    iteration: int = 0
    max_drawdown: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    avg_trade_length: float = 0.0
    feature_count: int = 0
    dominant_features: List[str] = field(default_factory=list)
    source_method: str = "manual"  # bayesian, genetic, ensemble, manual
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "threshold": self.threshold,
            "hold_period": self.hold_period,
            "sharpe": self.sharpe,
            "iteration": self.iteration,
            "max_drawdown": self.max_drawdown,
            "annual_return": self.annual_return,
            "volatility": self.volatility,
            "win_rate": self.win_rate,
            "avg_trade_length": self.avg_trade_length,
            "feature_count": self.feature_count,
            "dominant_features": self.dominant_features,
            "source_method": self.source_method,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(
            weights=d.get("weights", {}),
            threshold=d.get("threshold", 0.0),
            hold_period=d.get("hold_period", 0),
            sharpe=d.get("sharpe", 0.0),
            iteration=d.get("iteration", 0),
            max_drawdown=d.get("max_drawdown", 0.0),
            annual_return=d.get("annual_return", 0.0),
            volatility=d.get("volatility", 0.0),
            win_rate=d.get("win_rate", 0.0),
            avg_trade_length=d.get("avg_trade_length", 0.0),
            feature_count=d.get("feature_count", 0),
            dominant_features=d.get("dominant_features", []),
            source_method=d.get("source_method", "manual"),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class LearningMemory:
    """Enhanced memory system for learning patterns"""
    successful_patterns: Dict[str, Any] = field(default_factory=dict)
    failed_patterns: Dict[str, Any] = field(default_factory=dict)
    feature_success_rates: Dict[str, float] = field(default_factory=dict)
    phase_insights: Dict[str, List[str]] = field(default_factory=dict)
    improvement_history: List[Tuple[float, str]] = field(default_factory=list)
    convergence_signals: Dict[str, float] = field(default_factory=dict)
    
    def record_success_pattern(self, config: Config, pattern_name: str):
        """Record a successful configuration pattern"""
        if pattern_name not in self.successful_patterns:
            self.successful_patterns[pattern_name] = []
        self.successful_patterns[pattern_name].append(config.to_dict())
    
    def analyze_feature_performance(self, configs: List[Config]) -> Dict[str, float]:
        """Analyze which features lead to better performance"""
        feature_scores = {}
        for config in configs:
            for feature, weight in config.weights.items():
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(config.sharpe * abs(weight))
        
        # Calculate weighted average Sharpe for each feature
        for feature in feature_scores:
            feature_scores[feature] = sum(feature_scores[feature]) / len(feature_scores[feature])
        
        return feature_scores


@dataclass
class OptimizationState:
    """Enhanced optimization state with learning capabilities"""
    configs: List[Config] = field(default_factory=list)
    phase: ExplorationPhase = ExplorationPhase.FEATURE_DISCOVERY
    iteration: int = 0
    best_sharpe: float = float('-inf')
    best_config: Optional[Config] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    reflections: List[str] = field(default_factory=list)
    memory: LearningMemory = field(default_factory=LearningMemory)
    stagnation_count: int = 0
    last_improvement_iteration: int = 0
    adaptive_threshold: float = 0.01  # Minimum improvement to avoid stagnation
    
    def add_config(self, config: Config):
        """Add config with enhanced learning"""
        self.configs.append(config)
        
        # Track improvements
        improvement = config.sharpe - self.best_sharpe
        if improvement > self.adaptive_threshold:
            self.best_sharpe = config.sharpe
            self.best_config = config
            self.last_improvement_iteration = config.iteration
            self.stagnation_count = 0
            
            # Record successful pattern
            pattern = self._analyze_config_pattern(config)
            self.memory.record_success_pattern(config, pattern)
            self.memory.improvement_history.append((config.sharpe, pattern))
        else:
            self.stagnation_count += 1
        
        # Update feature importance based on recent performance
        self._update_feature_importance()
    
    def _analyze_config_pattern(self, config: Config) -> str:
        """Analyze the pattern of a successful configuration"""
        top_features = sorted(config.weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        feature_types = []
        
        for feature, _ in top_features:
            if "fed" in feature.lower() or "policy" in feature.lower():
                feature_types.append("fed")
            elif "vol" in feature.lower():
                feature_types.append("volatility")
            elif "mom" in feature.lower():
                feature_types.append("momentum")
            elif "rate" in feature.lower():
                feature_types.append("rates")
            else:
                feature_types.append("market")
        
        dominant_type = max(set(feature_types), key=feature_types.count)
        return f"{dominant_type}_{config.source_method}_{len(config.weights)}_features"
    
    def _update_feature_importance(self):
        """Update feature importance based on recent configs"""
        if len(self.configs) < 3:
            return
        
        # Use last 10 configs for adaptive learning
        recent_configs = self.configs[-10:]
        self.feature_importance = self.memory.analyze_feature_performance(recent_configs)
    
    def get_top_configs(self, n: int = 5) -> List[Config]:
        """Get top N configs by Sharpe"""
        sorted_configs = sorted(self.configs, key=lambda c: c.sharpe, reverse=True)
        return sorted_configs[:n]
    
    def is_stagnating(self, threshold_iterations: int = 5) -> bool:
        """Check if optimization is stagnating"""
        return self.stagnation_count >= threshold_iterations
    
    def get_convergence_signal(self) -> float:
        """Calculate convergence signal (0-1, higher = more converged)"""
        if len(self.configs) < 5:
            return 0.0
        
        recent_sharpes = [c.sharpe for c in self.configs[-5:]]
        variance = np.var(recent_sharpes)
        
        # Low variance = high convergence
        return max(0, 1 - variance * 10)
    
    def get_phase_for_iteration(self, iteration: int, max_iterations: int) -> ExplorationPhase:
        """Adaptive phase determination based on learning"""
        # Check if stagnating - if so, change phase earlier
        if self.is_stagnating():
            # Force phase change to break stagnation
            phases = list(ExplorationPhase)
            current_idx = phases.index(self.phase)
            next_idx = (current_idx + 1) % len(phases)
            return phases[next_idx]
        
        # Normal phase progression with convergence awareness
        convergence = self.get_convergence_signal()
        
        progress = iteration / max_iterations
        if progress < 0.20 and convergence < 0.7:
            return ExplorationPhase.FEATURE_DISCOVERY
        elif progress < 0.45 and convergence < 0.8:
            return ExplorationPhase.WEIGHT_OPTIMIZATION
        elif progress < 0.75 and convergence < 0.9:
            return ExplorationPhase.PARAMETER_TUNING
        else:
            return ExplorationPhase.REFINEMENT


class BayesianOptimizer:
    """
    Simple Bayesian-inspired optimizer.
    Uses a surrogate model to predict promising configurations.
    """
    
    def __init__(self, available_features: List[str]):
        self.available_features = available_features
        self.history: List[Config] = []
        
    def add_observation(self, config: Config):
        self.history.append(config)
    
    def suggest_next(self, n_suggestions: int = 3) -> List[Dict[str, Any]]:
        """
        Suggest next configurations to try based on history.
        Uses acquisition function to balance exploration/exploitation.
        """
        if len(self.history) < 3:
            # Not enough data - explore randomly
            return self._random_suggestions(n_suggestions)
        
        # Analyze which features appear in top configs
        top_configs = sorted(self.history, key=lambda c: c.sharpe, reverse=True)[:5]
        
        # Calculate feature importance from top configs
        feature_scores = {}
        for config in top_configs:
            for feat, weight in config.weights.items():
                if feat not in feature_scores:
                    feature_scores[feat] = []
                feature_scores[feat].append((weight, config.sharpe))
        
        # Calculate expected value for each feature
        feature_ev = {}
        for feat, scores in feature_scores.items():
            if scores:
                avg_weight = np.mean([s[0] for s in scores])
                avg_sharpe = np.mean([s[1] for s in scores])
                feature_ev[feat] = (avg_weight, avg_sharpe)
        
        # Generate suggestions based on best features + some exploration
        suggestions = []
        
        # 1. Exploitation: Refine best config
        if self.history:
            best = max(self.history, key=lambda c: c.sharpe)
            refined = self._mutate_config(best, intensity=0.1)
            suggestions.append(refined)
        
        # 2. Exploration: Try new feature combinations
        top_features = sorted(feature_ev.keys(), key=lambda f: feature_ev[f][1], reverse=True)[:6]
        for _ in range(min(2, n_suggestions - 1)):
            new_config = self._generate_from_features(
                random.sample(top_features, min(4, len(top_features)))
            )
            suggestions.append(new_config)
        
        return suggestions[:n_suggestions]
    
    def _random_suggestions(self, n: int) -> List[Dict[str, Any]]:
        """Generate random configurations for exploration"""
        suggestions = []
        for _ in range(n):
            n_features = random.randint(3, 6)
            features = random.sample(self.available_features, min(n_features, len(self.available_features)))
            config = self._generate_from_features(features)
            suggestions.append(config)
        return suggestions
    
    def _generate_from_features(self, features: List[str]) -> Dict[str, Any]:
        """Generate a config from a list of features"""
        weights = {}
        for feat in features:
            # Random weight between -0.3 and 0.3
            weights[feat] = round(random.uniform(-0.3, 0.3), 3)
        
        return {
            "weights": weights,
            "threshold": random.choice([0.0, 0.02, 0.04, 0.05, 0.1]),
            "hold_period": random.choice([0, 5, 10, 21, 42, 63]),
        }
    
    def _mutate_config(self, config: Config, intensity: float = 0.2) -> Dict[str, Any]:
        """Mutate a config slightly"""
        new_weights = {}
        for feat, weight in config.weights.items():
            # Add small perturbation
            delta = random.gauss(0, intensity * abs(weight) if weight != 0 else 0.05)
            new_weights[feat] = round(weight + delta, 4)
        
        # Maybe add or remove a feature
        if random.random() < 0.2 and self.available_features:
            unused = [f for f in self.available_features if f not in new_weights]
            if unused:
                new_feat = random.choice(unused)
                new_weights[new_feat] = round(random.uniform(-0.2, 0.2), 3)
        
        if random.random() < 0.1 and len(new_weights) > 2:
            feat_to_remove = random.choice(list(new_weights.keys()))
            del new_weights[feat_to_remove]
        
        return {
            "weights": new_weights,
            "threshold": config.threshold + random.choice([-0.01, 0, 0.01]),
            "hold_period": max(0, config.hold_period + random.choice([-5, 0, 5])),
        }


class GeneticOptimizer:
    """
    Genetic algorithm for config optimization.
    Treats configs as genomes that evolve through selection, crossover, and mutation.
    """
    
    def __init__(self, available_features: List[str], population_size: int = 10):
        self.available_features = available_features
        self.population_size = population_size
        self.population: List[Config] = []
        self.generation = 0
    
    def initialize_population(self, seed_configs: Optional[List[Config]] = None):
        """Initialize population with random or seeded configs"""
        self.population = []
        
        if seed_configs:
            self.population.extend(seed_configs[:self.population_size // 2])
        
        # Fill rest with random
        while len(self.population) < self.population_size:
            config = self._random_config()
            self.population.append(config)
    
    def evolve(self, fitness_scores: List[float]) -> List[Config]:
        """
        Evolve population based on fitness scores.
        Returns new generation of configs to test.
        """
        if len(fitness_scores) != len(self.population):
            raise ValueError("Fitness scores must match population size")
        
        # Update fitness
        for config, fitness in zip(self.population, fitness_scores):
            config.sharpe = fitness
        
        # Selection - keep top 50%
        sorted_pop = sorted(self.population, key=lambda c: c.sharpe, reverse=True)
        survivors = sorted_pop[:self.population_size // 2]
        
        # Crossover - breed survivors
        children = []
        while len(children) < self.population_size // 2:
            parent1, parent2 = random.sample(survivors, 2)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            children.append(child)
        
        self.population = survivors + children
        self.generation += 1
        
        return self.population
    
    def get_best(self) -> Optional[Config]:
        """Get best config in population"""
        if not self.population:
            return None
        return max(self.population, key=lambda c: c.sharpe)
    
    def _random_config(self) -> Config:
        """Generate random config"""
        n_features = random.randint(3, 6)
        features = random.sample(
            self.available_features, 
            min(n_features, len(self.available_features))
        )
        weights = {f: round(random.uniform(-0.3, 0.3), 3) for f in features}
        
        return Config(
            weights=weights,
            threshold=random.choice([0.0, 0.02, 0.04, 0.05, 0.1]),
            hold_period=random.choice([0, 5, 10, 21, 42, 63]),
        )
    
    def _crossover(self, parent1: Config, parent2: Config) -> Config:
        """Crossover two parent configs"""
        all_features = set(parent1.weights.keys()) | set(parent2.weights.keys())
        
        child_weights = {}
        for feat in all_features:
            if feat in parent1.weights and feat in parent2.weights:
                # Both parents have this feature - random choice
                child_weights[feat] = random.choice([
                    parent1.weights[feat], 
                    parent2.weights[feat]
                ])
            elif feat in parent1.weights:
                # Only parent1 has it - 50% chance to inherit
                if random.random() < 0.5:
                    child_weights[feat] = parent1.weights[feat]
            else:
                # Only parent2 has it
                if random.random() < 0.5:
                    child_weights[feat] = parent2.weights[feat]
        
        # Crossover parameters
        threshold = random.choice([parent1.threshold, parent2.threshold])
        hold_period = random.choice([parent1.hold_period, parent2.hold_period])
        
        return Config(
            weights=child_weights,
            threshold=threshold,
            hold_period=hold_period,
        )
    
    def _mutate(self, config: Config, mutation_rate: float = 0.2) -> Config:
        """Mutate a config"""
        new_weights = config.weights.copy()
        
        for feat in list(new_weights.keys()):
            if random.random() < mutation_rate:
                # Mutate weight
                delta = random.gauss(0, 0.05)
                new_weights[feat] = round(new_weights[feat] + delta, 4)
        
        # Maybe add new feature
        if random.random() < mutation_rate:
            unused = [f for f in self.available_features if f not in new_weights]
            if unused:
                new_feat = random.choice(unused)
                new_weights[new_feat] = round(random.uniform(-0.2, 0.2), 3)
        
        # Maybe remove feature
        if random.random() < mutation_rate / 2 and len(new_weights) > 2:
            feat_to_remove = random.choice(list(new_weights.keys()))
            del new_weights[feat_to_remove]
        
        # Mutate parameters
        new_threshold = config.threshold
        new_hold = config.hold_period
        
        if random.random() < mutation_rate:
            new_threshold = max(0, config.threshold + random.gauss(0, 0.02))
        if random.random() < mutation_rate:
            new_hold = max(0, config.hold_period + random.randint(-10, 10))
        
        return Config(
            weights=new_weights,
            threshold=round(new_threshold, 3),
            hold_period=new_hold,
        )


class EnsembleBuilder:
    """
    Combines top configs into an ensemble for more robust predictions.
    """
    
    @staticmethod
    def build_ensemble(
        configs: List[Config], 
        top_n: int = 5,
        method: str = "weighted_average"
    ) -> Dict[str, Any]:
        """
        Build ensemble from top configs.
        
        Methods:
        - weighted_average: Weight by Sharpe ratio
        - simple_average: Equal weight
        - best_features: Take best weight for each feature
        """
        if not configs:
            return {}
        
        top_configs = sorted(configs, key=lambda c: c.sharpe, reverse=True)[:top_n]
        
        if method == "weighted_average":
            # Normalize Sharpe ratios to weights
            sharpes = [max(0.01, c.sharpe) for c in top_configs]  # Avoid negative weights
            total_sharpe = sum(sharpes)
            config_weights = [s / total_sharpe for s in sharpes]
            
            # Weighted average of feature weights
            ensemble_weights = {}
            for config, cw in zip(top_configs, config_weights):
                for feat, weight in config.weights.items():
                    if feat not in ensemble_weights:
                        ensemble_weights[feat] = 0
                    ensemble_weights[feat] += cw * weight
            
            # Weighted average of parameters
            avg_threshold = sum(c.threshold * w for c, w in zip(top_configs, config_weights))
            avg_hold = sum(c.hold_period * w for c, w in zip(top_configs, config_weights))
            
        elif method == "simple_average":
            ensemble_weights = {}
            for config in top_configs:
                for feat, weight in config.weights.items():
                    if feat not in ensemble_weights:
                        ensemble_weights[feat] = []
                    ensemble_weights[feat].append(weight)
            
            ensemble_weights = {f: np.mean(w) for f, w in ensemble_weights.items()}
            avg_threshold = np.mean([c.threshold for c in top_configs])
            avg_hold = np.mean([c.hold_period for c in top_configs])
            
        elif method == "best_features":
            # For each feature, take weight from the config with highest Sharpe
            feature_best = {}
            for config in top_configs:
                for feat, weight in config.weights.items():
                    if feat not in feature_best or config.sharpe > feature_best[feat][1]:
                        feature_best[feat] = (weight, config.sharpe)
            
            ensemble_weights = {f: v[0] for f, v in feature_best.items()}
            avg_threshold = top_configs[0].threshold
            avg_hold = top_configs[0].hold_period
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Round values
        ensemble_weights = {f: round(w, 4) for f, w in ensemble_weights.items()}
        
        return {
            "weights": ensemble_weights,
            "threshold": round(avg_threshold, 3),
            "hold_period": int(avg_hold),
            "n_configs": len(top_configs),
            "method": method,
            "component_sharpes": [c.sharpe for c in top_configs],
        }


class SelfReflection:
    """
    Generates prompts for the model to reflect on what's working and why.
    """
    
    @staticmethod
    def generate_reflection_prompt(
        state: OptimizationState,
        last_n: int = 5
    ) -> str:
        """Generate a reflection prompt based on recent history"""
        
        if len(state.configs) < 2:
            return ""
        
        recent = state.configs[-last_n:]
        best = state.best_config
        
        # Find biggest improvements and regressions
        improvements = []
        regressions = []
        for i in range(1, len(recent)):
            delta = recent[i].sharpe - recent[i-1].sharpe
            if delta > 0.05:
                improvements.append((recent[i-1], recent[i], delta))
            elif delta < -0.05:
                regressions.append((recent[i-1], recent[i], delta))
        
        prompt = """
## Self-Reflection: Analyze What's Working

Before the next experiment, reflect on the optimization so far:

### Current State
- Best Sharpe: {best_sharpe:.3f} (iteration {best_iter})
- Configs tested: {n_configs}
- Current phase: {phase}

### Key Questions to Consider

1. **Feature Insights**: Which features consistently appear in top configs? Which hurt performance?
"""
        
        if best:
            prompt += f"""
2. **Best Config Analysis**: 
   The best config uses: {list(best.weights.keys())}
   - Why might these features predict {state.configs[0].iteration if state.configs else 'factor'} returns?
   - What's the economic intuition?

"""
        
        if improvements:
            prompt += """
3. **What Improved Performance**:
"""
            for prev, curr, delta in improvements[:2]:
                prompt += f"   - Sharpe +{delta:.3f}: Changed from {list(prev.weights.keys())} to {list(curr.weights.keys())}\n"
        
        if regressions:
            prompt += """
4. **What Hurt Performance**:
"""
            for prev, curr, delta in regressions[:2]:
                prompt += f"   - Sharpe {delta:.3f}: Changed from {list(prev.weights.keys())} to {list(curr.weights.keys())}\n"
        
        prompt += """
### Based on this reflection:
- What NEW approach should we try that we haven't tested yet?
- What patterns do you see in successful vs unsuccessful configs?

"""
        
        return prompt.format(
            best_sharpe=state.best_sharpe if state.best_sharpe > float('-inf') else 0,
            best_iter=best.iteration if best else 0,
            n_configs=len(state.configs),
            phase=state.phase.value,
        )
    
    @staticmethod
    def generate_phase_transition_prompt(
        old_phase: ExplorationPhase,
        new_phase: ExplorationPhase,
        state: OptimizationState
    ) -> str:
        """Generate prompt when transitioning between phases"""
        
        phase_instructions = {
            ExplorationPhase.FEATURE_DISCOVERY: """
## Phase: Feature Discovery

Focus on finding the BEST individual features:
- Test different feature combinations
- Use analyze_feature_correlations to find high-IC features
- Try features from different categories (rates, volatility, sectors)
- Don't worry about fine-tuning weights yet
""",
            ExplorationPhase.WEIGHT_OPTIMIZATION: """
## Phase: Weight Optimization

Now optimize the weights of our best features:
- Take the top features from Phase 1
- Try different weight combinations
- Test positive vs negative weights
- Use test_signal_combination with different weight values
""",
            ExplorationPhase.PARAMETER_TUNING: """
## Phase: Parameter Tuning

Fine-tune threshold and holding period:
- Use run_parameter_sweep to find optimal threshold (0, 0.02, 0.04, 0.05, 0.1)
- Test different holding periods (0, 5, 10, 21, 42, 63 days)
- Keep weights fixed from Phase 2
""",
            ExplorationPhase.REFINEMENT: """
## Phase: Refinement

Final fine-tuning:
- Make small adjustments to the best config
- Try adding/removing one feature at a time
- Test ensemble of top configs
- Validate on out-of-sample data
""",
        }
        
        prompt = f"""
{'='*60}
PHASE TRANSITION: {old_phase.value} → {new_phase.value}
{'='*60}

{phase_instructions.get(new_phase, "")}

### Summary from Previous Phase:
- Best Sharpe achieved: {state.best_sharpe:.3f}
- Top features: {list(state.best_config.weights.keys()) if state.best_config else []}

"""
        return prompt


class OutOfSampleValidator:
    """
    Validates configs on held-out data to catch overfitting.
    """
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """
        Split ratios:
        - train_ratio: For initial optimization
        - val_ratio: For validation during optimization
        - Remaining: Test (final evaluation only)
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
    
    def get_date_splits(
        self, 
        start_date: str = "2016-01-01",
        end_date: str = "2025-12-31"
    ) -> Dict[str, Tuple[str, str]]:
        """Get date ranges for train/val/test splits"""
        from datetime import datetime, timedelta
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end - start).days
        
        train_end = start + timedelta(days=int(total_days * self.train_ratio))
        val_end = train_end + timedelta(days=int(total_days * self.val_ratio))
        
        return {
            "train": (start_date, train_end.strftime("%Y-%m-%d")),
            "validation": (train_end.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")),
            "test": (val_end.strftime("%Y-%m-%d"), end_date),
        }
    
    def validate_config(
        self,
        config: Config,
        train_sharpe: float,
        val_sharpe: float
    ) -> Dict[str, Any]:
        """
        Validate a config by comparing train and validation performance.
        """
        sharpe_decay = train_sharpe - val_sharpe
        decay_ratio = sharpe_decay / abs(train_sharpe) if train_sharpe != 0 else 0
        
        # Flags for potential overfitting
        is_overfit = sharpe_decay > 0.3 or decay_ratio > 0.5
        is_robust = sharpe_decay < 0.1 and val_sharpe > 0.2
        
        return {
            "train_sharpe": train_sharpe,
            "val_sharpe": val_sharpe,
            "sharpe_decay": sharpe_decay,
            "decay_ratio": decay_ratio,
            "is_overfit": is_overfit,
            "is_robust": is_robust,
            "recommendation": "robust" if is_robust else "overfit" if is_overfit else "moderate",
        }


def get_phase_prompt(phase: ExplorationPhase, iteration: int, state: OptimizationState) -> str:
    """Get the appropriate prompt for the current phase"""
    
    base_prompt = f"""
Current Phase: {phase.value.upper()} (Iteration {iteration})
Best Sharpe so far: {state.best_sharpe:.3f}
"""
    
    if phase == ExplorationPhase.FEATURE_DISCOVERY:
        return base_prompt + """
GOAL: Find the best predictive features.
- Use analyze_feature_correlations to find high-IC features
- Test 3-5 different feature combinations
- Focus on finding features with |IC| > 0.05 and p-value < 0.1
"""
    
    elif phase == ExplorationPhase.WEIGHT_OPTIMIZATION:
        top_features = list(state.best_config.weights.keys()) if state.best_config else []
        return base_prompt + f"""
GOAL: Optimize weights for top features: {top_features}
- Try different weight combinations (-0.3 to +0.3)
- Test flipping signs
- Try adding one more feature from the discovery phase
"""
    
    elif phase == ExplorationPhase.PARAMETER_TUNING:
        return base_prompt + """
GOAL: Find optimal threshold and holding period.
- Use run_parameter_sweep with thresholds [0, 0.02, 0.04, 0.05, 0.1]
- Test holding periods [0, 5, 10, 21, 42, 63]
- Keep best weights from previous phase
"""
    
    elif phase == ExplorationPhase.REFINEMENT:
        return base_prompt + """
GOAL: Final refinement and validation.
- Make small adjustments to best config
- Validate on out-of-sample data
- Consider ensemble of top 3-5 configs
"""
    
    return base_prompt

