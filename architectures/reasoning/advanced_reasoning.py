"""
Advanced Reasoning Systems - SOTA

Implementations:
- Chain-of-Thought (CoT) prompting
- Tree of Thoughts (ToT)
- ReAct (Reasoning + Acting)
- Self-Consistency
- Program-of-Thoughts
- Tool Use / Function Calling
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class ReasoningStep:
    """Single step in reasoning chain"""

    def __init__(
        self,
        thought: str,
        action: Optional[str] = None,
        observation: Optional[str] = None
    ):
        self.thought = thought
        self.action = action
        self.observation = observation

    def __repr__(self):
        parts = [f"Thought: {self.thought}"]
        if self.action:
            parts.append(f"Action: {self.action}")
        if self.observation:
            parts.append(f"Observation: {self.observation}")
        return "\n".join(parts)


class ChainOfThought:
    """
    Chain-of-Thought reasoning.

    Generates intermediate reasoning steps before final answer.
    """

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_reasoning_chain(
        self,
        question: str,
        max_steps: int = 5,
        temperature: float = 0.7
    ) -> Tuple[List[str], str]:
        """
        Generate chain of reasoning steps.

        Args:
            question: Input question
            max_steps: Maximum reasoning steps
            temperature: Sampling temperature

        Returns:
            reasoning_steps: List of reasoning steps
            answer: Final answer
        """
        prompt = f"Question: {question}\nLet's think step by step.\n"

        reasoning_steps = []

        for step in range(max_steps):
            # Generate next reasoning step
            response = self._generate(prompt, temperature)

            # Check if we reached final answer
            if "answer is" in response.lower() or "therefore" in response.lower():
                reasoning_steps.append(response)
                # Extract answer
                answer = self._extract_answer(response)
                break

            reasoning_steps.append(response)
            prompt += f"Step {step + 1}: {response}\n"

        return reasoning_steps, answer

    def _generate(self, prompt: str, temperature: float) -> str:
        """Generate text from model"""
        # Placeholder - would use actual model
        return "Generated reasoning step"

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from reasoning"""
        # Simple extraction - would be more sophisticated
        if "answer is" in text.lower():
            return text.split("answer is")[-1].strip()
        return text


class TreeNode:
    """Node in Tree of Thoughts"""

    def __init__(
        self,
        state: str,
        parent: Optional['TreeNode'] = None,
        value: float = 0.0
    ):
        self.state = state
        self.parent = parent
        self.children: List[TreeNode] = []
        self.value = value
        self.visits = 0

    def add_child(self, child: 'TreeNode'):
        """Add child node"""
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Check if node is leaf"""
        return len(self.children) == 0

    def get_path(self) -> List[str]:
        """Get path from root to this node"""
        path = []
        node = self
        while node:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))


class TreeOfThoughts:
    """
    Tree of Thoughts reasoning.

    Explores multiple reasoning paths in tree structure.
    Uses search algorithm (BFS/DFS/MCTS) to find best solution.
    """

    def __init__(
        self,
        model: nn.Module,
        value_function: Callable[[str], float],
        breadth: int = 3,
        depth: int = 5
    ):
        self.model = model
        self.value_function = value_function
        self.breadth = breadth  # Number of thoughts per step
        self.depth = depth  # Maximum depth

    def solve(self, problem: str, search_type: str = 'bfs') -> TreeNode:
        """
        Solve problem using tree search.

        Args:
            problem: Problem description
            search_type: 'bfs', 'dfs', or 'mcts'

        Returns:
            Best solution node
        """
        root = TreeNode(state=f"Problem: {problem}")

        if search_type == 'bfs':
            return self._breadth_first_search(root)
        elif search_type == 'dfs':
            return self._depth_first_search(root)
        elif search_type == 'mcts':
            return self._monte_carlo_tree_search(root)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

    def _generate_thoughts(self, state: str, n: int) -> List[str]:
        """
        Generate n candidate thoughts from current state.

        Args:
            state: Current reasoning state
            n: Number of thoughts to generate

        Returns:
            List of thought candidates
        """
        # Placeholder - would use model to generate diverse thoughts
        thoughts = [
            f"Approach {i+1}: Consider {state} from angle {i+1}"
            for i in range(n)
        ]
        return thoughts

    def _breadth_first_search(self, root: TreeNode) -> TreeNode:
        """
        BFS to explore tree of thoughts.

        Returns best leaf node.
        """
        queue = [(root, 0)]  # (node, depth)
        best_node = root
        best_value = root.value

        while queue:
            node, depth = queue.pop(0)

            # Check if we reached max depth or solution
            if depth >= self.depth or self._is_solution(node.state):
                value = self.value_function(node.state)
                node.value = value

                if value > best_value:
                    best_value = value
                    best_node = node
                continue

            # Generate and evaluate thoughts
            thoughts = self._generate_thoughts(node.state, self.breadth)

            for thought in thoughts:
                child = TreeNode(state=thought)
                child.value = self.value_function(thought)
                node.add_child(child)
                queue.append((child, depth + 1))

        return best_node

    def _depth_first_search(self, root: TreeNode) -> TreeNode:
        """
        DFS to explore tree of thoughts.

        Returns best leaf node.
        """
        stack = [(root, 0)]
        best_node = root
        best_value = root.value

        while stack:
            node, depth = stack.pop()

            if depth >= self.depth or self._is_solution(node.state):
                value = self.value_function(node.state)
                node.value = value

                if value > best_value:
                    best_value = value
                    best_node = node
                continue

            # Generate thoughts
            thoughts = self._generate_thoughts(node.state, self.breadth)

            for thought in thoughts:
                child = TreeNode(state=thought)
                child.value = self.value_function(thought)
                node.add_child(child)
                stack.append((child, depth + 1))

        return best_node

    def _monte_carlo_tree_search(self, root: TreeNode, num_simulations: int = 100) -> TreeNode:
        """
        MCTS for tree of thoughts.

        More sophisticated than BFS/DFS.
        """
        for _ in range(num_simulations):
            # Selection
            node = self._select(root)

            # Expansion
            if not self._is_terminal(node):
                node = self._expand(node)

            # Simulation
            value = self._simulate(node)

            # Backpropagation
            self._backpropagate(node, value)

        # Return best child of root
        return max(root.children, key=lambda n: n.value / (n.visits + 1e-8))

    def _select(self, node: TreeNode) -> TreeNode:
        """Select most promising node using UCB"""
        while not node.is_leaf():
            node = max(
                node.children,
                key=lambda n: self._ucb(n)
            )
        return node

    def _ucb(self, node: TreeNode, c: float = 1.41) -> float:
        """Upper Confidence Bound for node selection"""
        if node.visits == 0:
            return float('inf')

        exploitation = node.value / node.visits
        exploration = c * (torch.tensor(node.parent.visits).log() / node.visits).sqrt()

        return exploitation + exploration

    def _expand(self, node: TreeNode) -> TreeNode:
        """Expand node with new children"""
        thoughts = self._generate_thoughts(node.state, self.breadth)

        for thought in thoughts:
            child = TreeNode(state=thought)
            node.add_child(child)

        return node.children[0] if node.children else node

    def _simulate(self, node: TreeNode) -> float:
        """Simulate rollout from node"""
        return self.value_function(node.state)

    def _backpropagate(self, node: TreeNode, value: float):
        """Backpropagate value up the tree"""
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    def _is_solution(self, state: str) -> bool:
        """Check if state is a solution"""
        # Placeholder - would check if reasoning reached conclusion
        return "final answer" in state.lower()

    def _is_terminal(self, node: TreeNode) -> bool:
        """Check if node is terminal"""
        return self._is_solution(node.state)


class ReAct:
    """
    ReAct: Synergizing Reasoning and Acting.

    Interleaves reasoning traces with task-specific actions.
    """

    def __init__(
        self,
        model: nn.Module,
        tools: Dict[str, Callable],
        max_steps: int = 10
    ):
        self.model = model
        self.tools = tools
        self.max_steps = max_steps

    def run(self, task: str) -> Tuple[List[ReasoningStep], str]:
        """
        Run ReAct loop.

        Args:
            task: Task description

        Returns:
            reasoning_chain: List of reasoning steps
            final_answer: Final answer
        """
        reasoning_chain = []
        context = f"Task: {task}\n"

        for step in range(self.max_steps):
            # Generate thought
            thought = self._generate_thought(context)
            print(f"Thought {step + 1}: {thought}")

            # Generate action
            action = self._generate_action(context, thought)
            print(f"Action {step + 1}: {action}")

            # Execute action and get observation
            observation = self._execute_action(action)
            print(f"Observation {step + 1}: {observation}")

            # Store step
            reasoning_step = ReasoningStep(thought, action, observation)
            reasoning_chain.append(reasoning_step)

            # Update context
            context += f"\nThought: {thought}\nAction: {action}\nObservation: {observation}\n"

            # Check if task is complete
            if self._is_complete(thought, observation):
                final_answer = self._extract_answer(thought, observation)
                break

        return reasoning_chain, final_answer

    def _generate_thought(self, context: str) -> str:
        """Generate reasoning thought"""
        # Placeholder - would use model
        return "I should search for information about X"

    def _generate_action(self, context: str, thought: str) -> str:
        """Generate action based on thought"""
        # Placeholder - would use model to select tool
        return "search[query]"

    def _execute_action(self, action: str) -> str:
        """Execute action using available tools"""
        # Parse action
        if '[' in action and ']' in action:
            tool_name = action.split('[')[0].strip()
            arg = action.split('[')[1].split(']')[0].strip()

            if tool_name in self.tools:
                return str(self.tools[tool_name](arg))

        return "Action not recognized"

    def _is_complete(self, thought: str, observation: str) -> bool:
        """Check if reasoning is complete"""
        return "answer is" in thought.lower() or "conclude" in thought.lower()

    def _extract_answer(self, thought: str, observation: str) -> str:
        """Extract final answer"""
        if "answer is" in thought.lower():
            return thought.split("answer is")[-1].strip()
        return observation


class SelfConsistency:
    """
    Self-Consistency reasoning.

    Samples multiple reasoning paths and takes majority vote.
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 5
    ):
        self.model = model
        self.num_samples = num_samples

    def solve(self, question: str) -> str:
        """
        Solve with self-consistency.

        Args:
            question: Input question

        Returns:
            Most consistent answer
        """
        # Generate multiple reasoning chains
        answers = []

        for _ in range(self.num_samples):
            # Generate with different sampling
            chain_of_thought = ChainOfThought(self.model, None)
            _, answer = chain_of_thought.generate_reasoning_chain(question)
            answers.append(answer)

        # Take majority vote
        final_answer = self._majority_vote(answers)

        return final_answer

    def _majority_vote(self, answers: List[str]) -> str:
        """Take majority vote among answers"""
        from collections import Counter

        # Normalize answers
        normalized = [ans.strip().lower() for ans in answers]

        # Count
        counter = Counter(normalized)
        most_common = counter.most_common(1)[0][0]

        # Return original casing
        for ans in answers:
            if ans.strip().lower() == most_common:
                return ans

        return answers[0]


class FunctionCallingSystem:
    """
    Function Calling / Tool Use system.

    Allows model to call external functions/APIs.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.functions: Dict[str, Callable] = {}

    def register_function(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        """
        Register a function for the model to call.

        Args:
            name: Function name
            function: Callable function
            description: Description of what function does
            parameters: JSON schema of parameters
        """
        self.functions[name] = {
            'function': function,
            'description': description,
            'parameters': parameters
        }

    def run(self, user_message: str) -> str:
        """
        Run with function calling.

        Args:
            user_message: User input

        Returns:
            Final response
        """
        messages = [{"role": "user", "content": user_message}]

        while True:
            # Generate response (with function call if needed)
            response = self._generate_with_functions(messages)

            # Check if model wants to call a function
            if 'function_call' in response:
                function_name = response['function_call']['name']
                arguments = json.loads(response['function_call']['arguments'])

                # Call function
                function_response = self._call_function(function_name, arguments)

                # Add to messages
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": str(function_response)
                })

            else:
                # No function call - return final response
                return response['content']

    def _generate_with_functions(self, messages: List[Dict]) -> Dict:
        """
        Generate response potentially with function call.

        Returns dict with 'content' or 'function_call'
        """
        # Placeholder - would use model with function calling
        # Example response:
        return {
            'function_call': {
                'name': 'search',
                'arguments': '{"query": "weather today"}'
            }
        }

    def _call_function(self, name: str, arguments: Dict) -> Any:
        """Call registered function"""
        if name not in self.functions:
            return {"error": f"Function {name} not found"}

        try:
            function = self.functions[name]['function']
            result = function(**arguments)
            return result
        except Exception as e:
            return {"error": str(e)}


# Example tool functions
def search_tool(query: str) -> str:
    """Search for information"""
    # Placeholder
    return f"Search results for: {query}"


def calculator_tool(expression: str) -> float:
    """Evaluate mathematical expression"""
    # Safe evaluation
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        return f"Error: {e}"


def weather_tool(location: str) -> Dict:
    """Get weather information"""
    # Placeholder
    return {
        "location": location,
        "temperature": "72°F",
        "condition": "Sunny"
    }
