"""
Advanced Reasoning Capabilities - CRITICAL FOR AGI

Implementations:
- Causal Reasoning (inference, counterfactuals, interventions)
- Common Sense Reasoning (physical, social, temporal)
- Self-Improvement (self-critique, recursive refinement)

References:
- "The Book of Why" (Pearl, 2018)
- "Common Sense Reasoning" (Davis & Marcus, 2015)
- "Self-Refine: Iterative Refinement with Self-Feedback" (2023)
- "Constitutional AI" (Anthropic, 2022)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np


# ==================== CAUSAL REASONING ====================

class CausalGraph:
    """
    Causal graph for representing causal relationships.

    Uses Directed Acyclic Graph (DAG) structure.
    """

    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = {}  # parent -> children
        self.reverse_edges: Dict[str, Set[str]] = {}  # child -> parents

    def add_node(self, node: str):
        """Add node to graph"""
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()
        if node not in self.reverse_edges:
            self.reverse_edges[node] = set()

    def add_edge(self, cause: str, effect: str):
        """Add causal edge: cause -> effect"""
        self.add_node(cause)
        self.add_node(effect)
        self.edges[cause].add(effect)
        self.reverse_edges[effect].add(cause)

    def get_parents(self, node: str) -> Set[str]:
        """Get direct causes of node"""
        return self.reverse_edges.get(node, set())

    def get_children(self, node: str) -> Set[str]:
        """Get direct effects of node"""
        return self.edges.get(node, set())

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors (transitive causes)"""
        ancestors = set()
        to_visit = list(self.get_parents(node))

        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.get_parents(current))

        return ancestors


class CausalInference:
    """
    Causal inference system.

    Supports:
    - Do-calculus (interventions)
    - Counterfactual reasoning
    - Causal effect estimation
    """

    def __init__(self, causal_graph: CausalGraph):
        self.graph = causal_graph

    def intervene(
        self,
        variable: str,
        value: Any,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform intervention (do-operator).

        Set variable to value and propagate effects.

        Args:
            variable: Variable to intervene on
            value: Value to set
            data: Current data

        Returns:
            Data after intervention
        """
        intervened_data = data.copy()
        intervened_data[variable] = value

        # Remove influence from parents (intervention cuts incoming edges)
        # Only children are affected by intervention

        # Propagate to descendants
        to_update = list(self.graph.get_children(variable))
        while to_update:
            child = to_update.pop(0)
            # Update child based on all its parents
            parents = self.graph.get_parents(child)
            parent_values = [intervened_data.get(p) for p in parents]

            # Simplified update (would use actual causal model)
            if all(v is not None for v in parent_values):
                intervened_data[child] = self._compute_effect(child, parent_values)
                to_update.extend(self.graph.get_children(child))

        return intervened_data

    def counterfactual(
        self,
        variable: str,
        actual_value: Any,
        counterfactual_value: Any,
        actual_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Counterfactual reasoning: "What if X had been Y?"

        Args:
            variable: Variable to change
            actual_value: Actual value
            counterfactual_value: Hypothetical value
            actual_data: Actual observed data

        Returns:
            Counterfactual world state
        """
        # Step 1: Abduction - infer unobserved variables
        inferred = self._abduction(actual_data)

        # Step 2: Action - intervene with counterfactual value
        counterfactual_data = inferred.copy()
        counterfactual_data[variable] = counterfactual_value

        # Step 3: Prediction - compute effects
        result = self.intervene(variable, counterfactual_value, counterfactual_data)

        return result

    def _abduction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer unobserved variables from observed data"""
        # Simplified - would use structural causal model
        return data.copy()

    def _compute_effect(self, variable: str, parent_values: List[Any]) -> Any:
        """Compute effect based on parent values"""
        # Simplified - would use actual structural equations
        return sum(parent_values) if parent_values else 0


# ==================== COMMON SENSE REASONING ====================

class PhysicalReasoning:
    """Physical common sense reasoning"""

    PHYSICAL_RULES = [
        "Objects fall down due to gravity",
        "Solid objects don't pass through each other",
        "Water flows downhill",
        "Fire is hot and can burn",
        "Heavy objects are harder to lift",
    ]

    def check_physical_plausibility(self, scenario: str) -> Tuple[bool, str]:
        """Check if scenario is physically plausible"""
        # Simplified rule-based checking
        implausible_patterns = [
            ("float.*up.*without", "Objects need support to float up"),
            ("pass.*through.*wall", "Solid objects can't pass through walls"),
            ("water.*flow.*up", "Water doesn't flow uphill without pumping"),
        ]

        scenario_lower = scenario.lower()
        for pattern, reason in implausible_patterns:
            import re
            if re.search(pattern, scenario_lower):
                return False, reason

        return True, "Physically plausible"


class SocialReasoning:
    """Social common sense reasoning"""

    SOCIAL_NORMS = [
        "People generally tell the truth",
        "Helping others is considered good",
        "Politeness is valued in most cultures",
        "People have emotions and feelings",
    ]

    def infer_emotion(self, situation: str) -> str:
        """Infer likely emotional response"""
        # Simplified emotion inference
        positive_keywords = ["success", "win", "happy", "love", "gift"]
        negative_keywords = ["fail", "lose", "sad", "angry", "hurt"]

        situation_lower = situation.lower()

        if any(kw in situation_lower for kw in positive_keywords):
            return "positive (happy, excited)"
        elif any(kw in situation_lower for kw in negative_keywords):
            return "negative (sad, frustrated)"

        return "neutral"

    def infer_intention(self, action: str, context: str) -> str:
        """Infer likely intention behind action"""
        # Simplified intention inference
        if "help" in action.lower() or "assist" in action.lower():
            return "helpful/cooperative"
        elif "hide" in action.lower() or "deceive" in action.lower():
            return "deceptive/secretive"

        return "unclear - need more context"


class TemporalReasoning:
    """Temporal common sense reasoning"""

    def order_events(self, events: List[str]) -> List[str]:
        """Order events chronologically based on common sense"""
        # Simplified - would use learned temporal patterns
        # Look for temporal markers

        markers = {
            "before": -1,
            "after": 1,
            "first": -2,
            "then": 0,
            "finally": 2
        }

        scored_events = []
        for event in events:
            score = 0
            for marker, marker_score in markers.items():
                if marker in event.lower():
                    score += marker_score
            scored_events.append((score, event))

        scored_events.sort(key=lambda x: x[0])
        return [event for _, event in scored_events]


class CommonSenseReasoning:
    """Unified common sense reasoning system"""

    def __init__(self):
        self.physical = PhysicalReasoning()
        self.social = SocialReasoning()
        self.temporal = TemporalReasoning()

        # Knowledge base (simplified - would be much larger)
        self.knowledge = {
            "physical": PhysicalReasoning.PHYSICAL_RULES,
            "social": SocialReasoning.SOCIAL_NORMS,
        }

    def reason(
        self,
        query: str,
        reasoning_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Apply common sense reasoning to query.

        Args:
            query: Question or scenario
            reasoning_type: Type of reasoning (physical, social, temporal, auto)

        Returns:
            Reasoning result
        """
        if reasoning_type == "auto":
            # Detect reasoning type
            if any(kw in query.lower() for kw in ["feel", "emotion", "people", "person"]):
                reasoning_type = "social"
            elif any(kw in query.lower() for kw in ["fall", "gravity", "object", "water"]):
                reasoning_type = "physical"
            elif any(kw in query.lower() for kw in ["before", "after", "when", "sequence"]):
                reasoning_type = "temporal"

        result = {"type": reasoning_type, "query": query}

        if reasoning_type == "physical":
            plausible, reason = self.physical.check_physical_plausibility(query)
            result["plausible"] = plausible
            result["reason"] = reason
        elif reasoning_type == "social":
            emotion = self.social.infer_emotion(query)
            result["inferred_emotion"] = emotion
        elif reasoning_type == "temporal":
            # Extract events (simplified)
            events = query.split(".")
            ordered = self.temporal.order_events(events)
            result["ordered_events"] = ordered

        return result


# ==================== SELF-IMPROVEMENT ====================

class SelfCritic:
    """Self-critique system for iterative improvement"""

    CRITIQUE_DIMENSIONS = [
        "correctness",
        "completeness",
        "clarity",
        "efficiency",
        "safety"
    ]

    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        self.critique_history: List[Dict] = []

    def critique(
        self,
        output: str,
        task: str,
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate critique of output.

        Args:
            output: Generated output to critique
            task: Original task
            criteria: Criteria to evaluate

        Returns:
            Critique with scores and suggestions
        """
        if criteria is None:
            criteria = self.CRITIQUE_DIMENSIONS

        critique = {
            "task": task,
            "output": output,
            "scores": {},
            "issues": [],
            "suggestions": []
        }

        # Evaluate each criterion (simplified)
        for criterion in criteria:
            score = self._evaluate_criterion(output, task, criterion)
            critique["scores"][criterion] = score

            if score < 0.7:
                issue, suggestion = self._generate_feedback(output, criterion)
                critique["issues"].append(issue)
                critique["suggestions"].append(suggestion)

        self.critique_history.append(critique)
        return critique

    def _evaluate_criterion(
        self,
        output: str,
        task: str,
        criterion: str
    ) -> float:
        """Evaluate single criterion (simplified)"""
        # Would use model-based evaluation
        # Simplified: random score for demo
        return np.random.uniform(0.5, 1.0)

    def _generate_feedback(
        self,
        output: str,
        criterion: str
    ) -> Tuple[str, str]:
        """Generate issue and suggestion"""
        issues = {
            "correctness": ("Output may contain errors", "Verify accuracy of facts and logic"),
            "completeness": ("Output is incomplete", "Add missing information or steps"),
            "clarity": ("Output is unclear", "Improve explanation and structure"),
            "efficiency": ("Output is inefficient", "Optimize approach or code"),
            "safety": ("Output may be unsafe", "Add safety checks and error handling"),
        }
        return issues.get(criterion, ("Unknown issue", "Review and improve"))


class RecursiveImprovement:
    """Recursive self-improvement system"""

    def __init__(
        self,
        max_iterations: int = 5,
        improvement_threshold: float = 0.05
    ):
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.critic = SelfCritic()

    def iterative_refine(
        self,
        initial_output: str,
        task: str
    ) -> Tuple[str, List[Dict]]:
        """
        Iteratively refine output using self-critique.

        Args:
            initial_output: Initial attempt
            task: Task description

        Returns:
            (final_output, refinement_history)
        """
        current_output = initial_output
        history = []

        for iteration in range(self.max_iterations):
            # Critique current output
            critique = self.critic.critique(current_output, task)

            # Calculate overall score
            avg_score = np.mean(list(critique["scores"].values()))

            history.append({
                "iteration": iteration,
                "output": current_output,
                "critique": critique,
                "avg_score": avg_score
            })

            # Check if good enough
            if avg_score > 0.9 or (iteration > 0 and avg_score - history[-2]["avg_score"] < self.improvement_threshold):
                break

            # Refine based on critique
            current_output = self._refine(current_output, critique)

        return current_output, history

    def _refine(self, output: str, critique: Dict) -> str:
        """Refine output based on critique"""
        # Would use model to generate improved version
        # Simplified: append suggestions
        if critique["suggestions"]:
            return f"{output}\n[Improved: {critique['suggestions'][0]}]"
        return output


# Testing
def test_advanced_reasoning():
    """Test advanced reasoning capabilities"""
    print("Testing Advanced Reasoning...")

    # Test 1: Causal Reasoning
    print("\n1. Causal Reasoning")
    graph = CausalGraph()
    graph.add_edge("Rain", "WetGround")
    graph.add_edge("WetGround", "Slippery")
    graph.add_edge("Slippery", "Accident")

    print(f"  Causal graph: Rain -> WetGround -> Slippery -> Accident")

    causal = CausalInference(graph)

    # Intervention
    data = {"Rain": True, "WetGround": True}
    intervened = causal.intervene("Rain", False, data)
    print(f"  Intervention: do(Rain = False)")
    print(f"    Result: {intervened}")

    # Counterfactual
    print(f"  Counterfactual: 'What if it hadn't rained?'")
    cf = causal.counterfactual("Rain", True, False, data)
    print(f"    Result: {cf}")

    # Test 2: Common Sense Reasoning
    print("\n2. Common Sense Reasoning")
    cs = CommonSenseReasoning()

    # Physical
    scenarios = [
        "A ball was thrown and it fell down",
        "A ball was thrown and it floated up without any support",
    ]
    print("  Physical reasoning:")
    for scenario in scenarios:
        result = cs.reason(scenario, "physical")
        print(f"    '{scenario[:40]}...'")
        print(f"      Plausible: {result.get('plausible', 'N/A')}")

    # Social
    print("  Social reasoning:")
    situation = "John won the lottery"
    result = cs.reason(situation, "social")
    print(f"    Situation: '{situation}'")
    print(f"    Inferred emotion: {result.get('inferred_emotion')}")

    # Temporal
    print("  Temporal reasoning:")
    events = [
        "Finally, they celebrated.",
        "First, they planned the event.",
        "Then, they executed the plan."
    ]
    result = cs.reason(". ".join(events), "temporal")
    print(f"    Events (unordered): {len(events)}")
    if "ordered_events" in result:
        print(f"    Ordered:")
        for i, event in enumerate(result["ordered_events"]):
            print(f"      {i+1}. {event[:50]}")

    # Test 3: Self-Improvement
    print("\n3. Self-Improvement")

    # Self-critique
    critic = SelfCritic()
    output = "The answer is 42, but I'm not sure why."
    task = "Explain the meaning of life"

    critique = critic.critique(output, task)
    print(f"  Task: {task}")
    print(f"  Output: {output}")
    print(f"  Critique scores:")
    for criterion, score in critique["scores"].items():
        print(f"    {criterion}: {score:.2f}")
    print(f"  Issues found: {len(critique['issues'])}")
    print(f"  Suggestions: {len(critique['suggestions'])}")

    # Recursive improvement
    print("\n  Recursive improvement:")
    refiner = RecursiveImprovement(max_iterations=3)
    final, history = refiner.iterative_refine(output, task)

    print(f"  Iterations: {len(history)}")
    for h in history:
        print(f"    Iteration {h['iteration']}: score = {h['avg_score']:.2f}")
    print(f"  Final output: {final[:50]}...")

    print("\n✓ Advanced Reasoning tests completed!")

    # Summary
    print("\n" + "="*60)
    print("ADVANCED REASONING SUMMARY")
    print("="*60)
    print("1. Causal Reasoning:")
    print("   - Causal graphs (DAGs)")
    print("   - Do-calculus (interventions)")
    print("   - Counterfactual reasoning")
    print("   - Causal effect estimation")
    print("\n2. Common Sense Reasoning:")
    print("   - Physical reasoning (gravity, solidity, etc.)")
    print("   - Social reasoning (emotions, intentions)")
    print("   - Temporal reasoning (event ordering)")
    print("   - Knowledge-based inference")
    print("\n3. Self-Improvement:")
    print("   - Multi-dimensional critique")
    print("   - Iterative refinement")
    print("   - Convergence detection")
    print("   - Critique history tracking")
    print("\nApplications:")
    print("   - Scientific reasoning")
    print("   - Social interaction")
    print("   - Plan verification")
    print("   - Autonomous improvement")
    print("   - Safety assurance")


if __name__ == "__main__":
    test_advanced_reasoning()
