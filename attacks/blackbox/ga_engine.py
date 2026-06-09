"""
attacks/blackbox/ga_engine.py

Redesigned Genetic Algorithm for realistic black-box adversarial attack.

Key improvements over original
--------------------------------
1. Fitness tracks perturbation from SEED (not from zero-vector).
2. Multi-objective: evasion + minimal perturbation + traffic realism.
3. Oracle is injected → testable and thread-safe.
4. No top-level side effects; all setup is inside run_attack().
5. Adaptive mutation scale: starts aggressive, anneals over generations.
6. Elitism preserved across generations.
7. Verbose logging with per-generation stats.
8. Returns a rich result dict including query efficiency metrics.
"""

from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from deap import base, creator, tools

from attacks.blackbox.constraints import repair, realism_penalty
from attacks.blackbox.fitness import evaluate
from attacks.blackbox.genome import get_feature_names, get_attack_seeds
from attacks.blackbox.oracle import EnsembleOracle
from core.config import GA_CONFIG


# ─── DEAP setup (idempotent) ─────────────────────────────────────────────────

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class AttackResult:
    success: bool
    generation: int
    queries: int
    fitness: float
    probability: float
    perturbation_norm: float
    realism_penalty: float
    adversarial_sample: Optional[List[float]]
    seed_sample: List[float]
    feature_names: List[str]
    generation_log: List[dict] = field(default_factory=list)

    def query_efficiency(self) -> float:
        """Fraction of budget consumed."""
        limit = GA_CONFIG["query_limit"]
        return self.queries / limit

    def summary(self) -> str:
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        return (
            f"{status} | gen={self.generation} | "
            f"queries={self.queries} | prob={self.probability:.4f} | "
            f"perturbation={self.perturbation_norm:.4f} | "
            f"realism_penalty={self.realism_penalty:.4f}"
        )


# ─── Main attack ─────────────────────────────────────────────────────────────

def run_attack(
    oracle: Optional[EnsembleOracle] = None,
    seed_idx: Optional[int] = None,
    verbose: bool = True,
) -> AttackResult:
    """
    Run the black-box genetic attack.

    Parameters
    ----------
    oracle   : EnsembleOracle instance.  If None, a fresh one is created.
    seed_idx : Index into attack_seeds to use as starting point.
               If None, a random attack sample is chosen.
    verbose  : Print per-generation stats.

    Returns
    -------
    AttackResult with full diagnostics.
    """
    if oracle is None:
        oracle = EnsembleOracle(GA_CONFIG["query_limit"])
    oracle.reset()

    feature_names  = get_feature_names()
    attack_seeds   = get_attack_seeds()
    n_features     = len(feature_names)

    cfg  = GA_CONFIG
    POP  = cfg["population_size"]
    GENS = cfg["generations"]
    ELIT = cfg["elite_size"]
    CXPB = cfg["crossover_prob"]
    MUPB = cfg["mutation_prob"]
    GPMT = cfg["gene_mutation_prob"]
    EVASION_THRESH = cfg["evasion_threshold"]

    # ── Choose seed ───────────────────────────────────────────────────────────
    if seed_idx is None:
        seed_idx = random.randrange(len(attack_seeds))
    original_seed = list(attack_seeds[seed_idx])

    if verbose:
        print(f"\n[GA] Attack started | seed_idx={seed_idx} | "
              f"features={n_features} | pop={POP} | gens={GENS}")

    # ── Toolbox setup ─────────────────────────────────────────────────────────
    toolbox = base.Toolbox()

    def _make_individual():
        # Small Gaussian perturbation around seed to initialize population
        noise = np.random.normal(0, 0.01, n_features)
        ind = [s + n for s, n in zip(original_seed, noise)]
        ind = repair(ind, feature_names)
        return creator.Individual(ind)

    toolbox.register("individual",  _make_individual)
    toolbox.register("population",  tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate",        tools.cxBlend, alpha=0.3)
    toolbox.register("select",      tools.selTournament, tournsize=3)

    # ── Adaptive mutation ────────────────────────────────────────────────────
    def _mutate(individual, scale: float):
        for i in range(n_features):
            if random.random() < GPMT:
                v = individual[i]
                individual[i] += random.gauss(0, max(abs(v) * scale, 0.001))
        return individual,

    # ── Evaluate wrapper ─────────────────────────────────────────────────────
    def _eval(individual):
        repaired = repair(list(individual), feature_names)
        individual[:] = repaired
        if oracle.budget_exceeded():
            return (-999.0,)
        fit, _, _, _ = evaluate(individual, original_seed, feature_names, oracle)
        return (fit,)

    toolbox.register("evaluate", _eval)

    # ── Initialize population ────────────────────────────────────────────────
    population = toolbox.population(n=POP)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    best_ever: Optional[creator.Individual] = None
    best_fitness = -999.0
    gen_log = []

    # ── Evolution loop ────────────────────────────────────────────────────────
    for gen in range(GENS):
        if oracle.budget_exceeded():
            break

        # Adaptive mutation scale: anneal from 0.10 → 0.01
        scale = 0.10 * (1.0 - gen / GENS) + 0.01

        # Evaluate population
        invalid = [ind for ind in population if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Track best
        current_best = tools.selBest(population, 1)[0]
        cb_fit, cb_pred, cb_prob, cb_pert = evaluate(
            list(current_best), original_seed, feature_names, oracle
        )

        if cb_fit > best_fitness:
            best_fitness = cb_fit
            best_ever    = toolbox.clone(current_best)

        log_entry = {
            "generation":  gen + 1,
            "fitness":     cb_fit,
            "probability": cb_prob,
            "prediction":  cb_pred,
            "perturbation": cb_pert,
            "queries":     oracle.query_count,
        }
        gen_log.append(log_entry)

        if verbose:
            print(
                f"Gen {gen+1:03d} | fit={cb_fit:+.4f} | "
                f"prob={cb_prob:.4f} | pred={'Benign' if cb_pred==0 else 'Attack'} | "
                f"pert={cb_pert:.4f} | Q={oracle.query_count}"
            )

        # ── Early exit on success ─────────────────────────────────────────────
        if cb_prob < EVASION_THRESH:
            realism = realism_penalty(list(current_best), feature_names)
            if verbose:
                print(f"\n[GA] ATTACK SUCCESS at generation {gen+1}!")
            return AttackResult(
                success=True,
                generation=gen + 1,
                queries=oracle.query_count,
                fitness=cb_fit,
                probability=cb_prob,
                perturbation_norm=cb_pert,
                realism_penalty=realism,
                adversarial_sample=list(current_best),
                seed_sample=original_seed,
                feature_names=feature_names,
                generation_log=gen_log,
            )

        # ── Selection + reproduction ──────────────────────────────────────────
        elites   = tools.selBest(population, ELIT)
        offspring = list(map(toolbox.clone, toolbox.select(population, POP - ELIT)))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
                c1[:] = repair(c1, feature_names)
                c2[:] = repair(c2, feature_names)
                del c1.fitness.values
                del c2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUPB:
                _mutate(mutant, scale)
                mutant[:] = repair(mutant, feature_names)
                if mutant.fitness.valid:
                    del mutant.fitness.values

        population[:] = list(map(toolbox.clone, elites)) + offspring

    # ── Attack failed ─────────────────────────────────────────────────────────
    final_prob = gen_log[-1]["probability"] if gen_log else 1.0
    final_pert = gen_log[-1]["perturbation"] if gen_log else 0.0
    realism    = realism_penalty(list(best_ever) if best_ever else original_seed,
                                 feature_names)

    if verbose:
        print(f"\n[GA] Attack failed after {GENS} generations.")

    return AttackResult(
        success=False,
        generation=GENS,
        queries=oracle.query_count,
        fitness=best_fitness,
        probability=final_prob,
        perturbation_norm=final_pert,
        realism_penalty=realism,
        adversarial_sample=list(best_ever) if best_ever else None,
        seed_sample=original_seed,
        feature_names=feature_names,
        generation_log=gen_log,
    )
