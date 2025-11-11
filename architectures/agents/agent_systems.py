"""
Agent Systems - Tool Use, Multi-Agent, Planning, Memory

Enables autonomous agents that can use tools, collaborate, plan, and remember.

Key Components:
- Tool Use: Function calling and execution
- Multi-Agent: Collaboration and communication
- Planning: Goal decomposition and task planning
- Memory: Short-term and long-term memory systems

References:
- Toolformer: https://arxiv.org/abs/2302.04761
- ReAct: https://arxiv.org/abs/2210.03629
- AutoGPT: https://github.com/Significant-Gravitas/AutoGPT
- MemGPT: https://arxiv.org/abs/2310.08560
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from enum import Enum
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import json


# ============================================================================
# Tool Use
# ============================================================================

class ToolParameter:
    """Parameter definition for a tool"""
    def __init__(
        self,
        name: str,
        param_type: str,
        description: str,
        required: bool = True,
        enum: Optional[List[Any]] = None
    ):
        self.name = name
        self.param_type = param_type
        self.description = description
        self.required = required
        self.enum = enum


@dataclass
class ToolDefinition:
    """
    Definition of a tool that an agent can use.

    Example:
        >>> calculator = ToolDefinition(
        ...     name="calculator",
        ...     description="Perform mathematical calculations",
        ...     parameters=[
        ...         ToolParameter("expression", "string", "Math expression to evaluate")
        ...     ],
        ...     function=lambda expression: eval(expression)
        ... )
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.param_type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        return self.function(**kwargs)


class ToolRegistry:
    """
    Registry of available tools.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(calculator_tool)
        >>> registry.register(search_tool)
        >>>
        >>> # Agent can query available tools
        >>> tools = registry.get_all_tools()
    """

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        """Register a tool"""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get tool by name"""
        return self.tools.get(name)

    def get_all_tools(self) -> List[ToolDefinition]:
        """Get all registered tools"""
        return list(self.tools.values())

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools"""
        descriptions = []
        for tool in self.tools.values():
            params = ", ".join([p.name for p in tool.parameters])
            descriptions.append(
                f"{tool.name}({params}): {tool.description}"
            )
        return "\n".join(descriptions)


class ToolUseAgent:
    """
    Agent that can use tools via function calling.

    Example:
        >>> # Set up tools
        >>> registry = ToolRegistry()
        >>> registry.register(calculator)
        >>> registry.register(search)
        >>>
        >>> # Create agent
        >>> agent = ToolUseAgent(model, registry)
        >>>
        >>> # Agent uses tools to answer questions
        >>> response = agent.run(
        ...     "What is the population of Tokyo times 2?"
        ... )
        >>> # Agent will:
        >>> # 1. Call search("population of Tokyo") -> "14 million"
        >>> # 2. Call calculator("14000000 * 2") -> "28000000"
        >>> # 3. Return "28 million"
    """

    def __init__(
        self,
        model: nn.Module,
        tool_registry: ToolRegistry,
        max_iterations: int = 10
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations

    def run(
        self,
        query: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run agent on query.

        Returns:
            Dict with 'response', 'tool_calls', 'trajectory'
        """
        tool_calls = []
        trajectory = []

        # Build system prompt with tool descriptions
        system_prompt = f"""You are a helpful assistant with access to the following tools:

{self.tool_registry.get_tool_descriptions()}

To use a tool, output: TOOL[tool_name](param1=value1, param2=value2)
"""

        context = system_prompt + f"\n\nUser: {query}\nAssistant:"

        for iteration in range(self.max_iterations):
            # Generate response
            response = self._generate(context)

            if verbose:
                print(f"\nIteration {iteration + 1}:")
                print(f"Response: {response}")

            # Check if tool call
            if "TOOL[" in response:
                tool_call = self._parse_tool_call(response)

                if tool_call:
                    tool_name, params = tool_call
                    tool = self.tool_registry.get(tool_name)

                    if tool:
                        # Execute tool
                        try:
                            result = tool.execute(**params)
                            tool_calls.append({
                                'tool': tool_name,
                                'params': params,
                                'result': result
                            })

                            # Update context
                            context += f"\n{response}\nResult: {result}\nAssistant:"

                            trajectory.append({
                                'type': 'tool_call',
                                'tool': tool_name,
                                'params': params,
                                'result': result
                            })

                            if verbose:
                                print(f"Tool: {tool_name}")
                                print(f"Result: {result}")

                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            context += f"\n{response}\n{error_msg}\nAssistant:"

                            trajectory.append({
                                'type': 'error',
                                'error': error_msg
                            })
                    else:
                        error_msg = f"Unknown tool: {tool_name}"
                        context += f"\n{response}\n{error_msg}\nAssistant:"
                else:
                    # Failed to parse tool call
                    break
            else:
                # Final answer
                trajectory.append({
                    'type': 'final_answer',
                    'response': response
                })
                break

        return {
            'response': response,
            'tool_calls': tool_calls,
            'trajectory': trajectory
        }

    def _generate(self, prompt: str) -> str:
        """Generate response from model"""
        # In practice, use actual model generation
        return f"Generated response to: {prompt[:50]}..."

    def _parse_tool_call(self, response: str) -> Optional[Tuple[str, Dict]]:
        """Parse tool call from response"""
        # Simple parser for TOOL[name](param=value)
        try:
            start = response.index("TOOL[") + 5
            end = response.index("]", start)
            tool_name = response[start:end]

            params_start = response.index("(", end) + 1
            params_end = response.index(")", params_start)
            params_str = response[params_start:params_end]

            # Parse parameters
            params = {}
            if params_str.strip():
                for param in params_str.split(","):
                    key, value = param.split("=")
                    params[key.strip()] = value.strip()

            return tool_name, params

        except Exception:
            return None


# ============================================================================
# Multi-Agent System
# ============================================================================

class AgentRole(Enum):
    """Roles for multi-agent systems"""
    LEADER = "leader"
    WORKER = "worker"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"


@dataclass
class Message:
    """Message between agents"""
    sender: str
    recipient: str
    content: str
    message_type: str = "text"  # "text", "tool_result", "task"


class Agent(ABC):
    """Base class for agents in multi-agent system"""

    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role
        self.inbox: List[Message] = []

    def send_message(self, recipient: str, content: str, message_type: str = "text"):
        """Send message to another agent"""
        message = Message(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=message_type
        )
        return message

    def receive_message(self, message: Message):
        """Receive message"""
        self.inbox.append(message)

    @abstractmethod
    def process_messages(self) -> List[Message]:
        """Process inbox and generate responses"""
        pass


class LeaderAgent(Agent):
    """
    Leader agent that coordinates other agents.

    Responsibilities:
    - Task decomposition
    - Work assignment
    - Result aggregation
    """

    def __init__(self, name: str = "Leader"):
        super().__init__(name, AgentRole.LEADER)
        self.workers: List[str] = []

    def decompose_task(self, task: str) -> List[str]:
        """Decompose task into subtasks"""
        # In practice, use LLM to decompose
        subtasks = [
            f"Subtask 1 of: {task}",
            f"Subtask 2 of: {task}",
            f"Subtask 3 of: {task}"
        ]
        return subtasks

    def process_messages(self) -> List[Message]:
        """Process messages and coordinate"""
        responses = []

        for message in self.inbox:
            if message.message_type == "task":
                # Decompose and assign
                subtasks = self.decompose_task(message.content)

                for i, subtask in enumerate(subtasks):
                    worker = self.workers[i % len(self.workers)]
                    response = self.send_message(worker, subtask, "task")
                    responses.append(response)

        self.inbox.clear()
        return responses


class WorkerAgent(Agent):
    """
    Worker agent that executes tasks.

    Responsibilities:
    - Execute assigned tasks
    - Report results to leader
    """

    def __init__(self, name: str, tools: Optional[ToolRegistry] = None):
        super().__init__(name, AgentRole.WORKER)
        self.tools = tools or ToolRegistry()

    def execute_task(self, task: str) -> str:
        """Execute a task"""
        # In practice, use LLM + tools
        return f"Completed: {task}"

    def process_messages(self) -> List[Message]:
        """Process tasks from leader"""
        responses = []

        for message in self.inbox:
            if message.message_type == "task":
                # Execute task
                result = self.execute_task(message.content)
                response = self.send_message(message.sender, result, "result")
                responses.append(response)

        self.inbox.clear()
        return responses


class MultiAgentSystem:
    """
    Multi-agent system coordinator.

    Example:
        >>> # Create multi-agent system
        >>> mas = MultiAgentSystem()
        >>>
        >>> # Add agents
        >>> leader = LeaderAgent("Leader")
        >>> worker1 = WorkerAgent("Worker1")
        >>> worker2 = WorkerAgent("Worker2")
        >>>
        >>> mas.add_agent(leader)
        >>> mas.add_agent(worker1)
        >>> mas.add_agent(worker2)
        >>>
        >>> leader.workers = ["Worker1", "Worker2"]
        >>>
        >>> # Run task
        >>> result = mas.run("Build a website")
        >>> # Leader decomposes, workers execute, leader aggregates
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def add_agent(self, agent: Agent):
        """Add agent to system"""
        self.agents[agent.name] = agent

    def route_messages(self, messages: List[Message]):
        """Route messages to recipients"""
        for message in messages:
            recipient = self.agents.get(message.recipient)
            if recipient:
                recipient.receive_message(message)

    def run(
        self,
        task: str,
        max_rounds: int = 10
    ) -> Dict[str, Any]:
        """
        Run multi-agent system on task.

        Returns:
            Results from all agents
        """
        # Send initial task to leader
        if "Leader" in self.agents:
            leader = self.agents["Leader"]
            initial_message = Message(
                sender="System",
                recipient="Leader",
                content=task,
                message_type="task"
            )
            leader.receive_message(initial_message)

        # Run rounds
        all_messages = []

        for round_idx in range(max_rounds):
            round_messages = []

            # Each agent processes its inbox
            for agent in self.agents.values():
                messages = agent.process_messages()
                round_messages.extend(messages)

            # Route messages
            self.route_messages(round_messages)

            all_messages.extend(round_messages)

            # Check if done (no more messages)
            if not round_messages:
                break

        return {
            'rounds': round_idx + 1,
            'messages': all_messages
        }


# ============================================================================
# Planning
# ============================================================================

@dataclass
class Goal:
    """Goal to achieve"""
    description: str
    success_criteria: str
    completed: bool = False


@dataclass
class Plan:
    """Plan to achieve goal"""
    goal: Goal
    steps: List[str]
    current_step: int = 0


class Planner:
    """
    Hierarchical planner for goal decomposition.

    Example:
        >>> planner = Planner(model)
        >>> goal = Goal(
        ...     description="Build a website",
        ...     success_criteria="Website is live and functional"
        ... )
        >>> plan = planner.create_plan(goal)
        >>> # Returns: ["Design layout", "Write HTML", "Deploy", ...]
        >>>
        >>> # Execute plan
        >>> for step in plan.steps:
        ...     execute(step)
        ...     plan.current_step += 1
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def create_plan(self, goal: Goal) -> Plan:
        """
        Create hierarchical plan for goal.

        Uses chain-of-thought prompting to decompose goal.
        """
        prompt = f"""Create a step-by-step plan to achieve the following goal:

Goal: {goal.description}
Success criteria: {goal.success_criteria}

Break down the goal into concrete, executable steps:
"""
        # In practice, use LLM to generate plan
        steps = [
            "Research requirements",
            "Design solution",
            "Implement core functionality",
            "Test thoroughly",
            "Deploy and monitor"
        ]

        return Plan(goal=goal, steps=steps)

    def refine_plan(self, plan: Plan, feedback: str) -> Plan:
        """Refine plan based on feedback"""
        # In practice, use LLM to adjust plan
        return plan

    def check_progress(self, plan: Plan) -> Dict[str, Any]:
        """Check progress on plan"""
        total_steps = len(plan.steps)
        completed_steps = plan.current_step
        progress = completed_steps / total_steps if total_steps > 0 else 0

        return {
            'progress': progress,
            'completed_steps': completed_steps,
            'total_steps': total_steps,
            'current_step': plan.steps[plan.current_step] if plan.current_step < total_steps else None
        }


# ============================================================================
# Memory Systems
# ============================================================================

@dataclass
class MemoryEntry:
    """Single memory entry"""
    content: str
    timestamp: float
    importance: float = 0.5
    access_count: int = 0
    embedding: Optional[torch.Tensor] = None


class ShortTermMemory:
    """
    Short-term memory (working memory).

    Similar to KV cache in Transformers.

    Example:
        >>> stm = ShortTermMemory(capacity=10)
        >>> stm.add("User likes pizza")
        >>> stm.add("User is in New York")
        >>>
        >>> # Query memory
        >>> context = stm.get_context()
        >>> # "User likes pizza. User is in New York."
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.memories: List[MemoryEntry] = []

    def add(self, content: str, importance: float = 0.5):
        """Add memory"""
        memory = MemoryEntry(
            content=content,
            timestamp=0.0,  # In practice, use time.time()
            importance=importance
        )
        self.memories.append(memory)

        # Evict if over capacity
        if len(self.memories) > self.capacity:
            # Evict least important
            self.memories.sort(key=lambda m: m.importance)
            self.memories = self.memories[1:]

    def get_context(self) -> str:
        """Get context string from memories"""
        return " ".join([m.content for m in self.memories])

    def clear(self):
        """Clear all memories"""
        self.memories.clear()


class LongTermMemory:
    """
    Long-term memory with semantic search.

    Stores memories with embeddings for retrieval.

    Example:
        >>> ltm = LongTermMemory(embedding_model)
        >>> ltm.add("Paris is the capital of France")
        >>> ltm.add("Tokyo is the capital of Japan")
        >>>
        >>> # Query by similarity
        >>> results = ltm.query("What is the capital of France?", k=1)
        >>> # Returns: "Paris is the capital of France"
    """

    def __init__(self, embedding_model: Optional[nn.Module] = None):
        self.embedding_model = embedding_model
        self.memories: List[MemoryEntry] = []

    def add(self, content: str, importance: float = 0.5):
        """Add memory with embedding"""
        # Compute embedding
        if self.embedding_model:
            embedding = self._compute_embedding(content)
        else:
            embedding = None

        memory = MemoryEntry(
            content=content,
            timestamp=0.0,
            importance=importance,
            embedding=embedding
        )
        self.memories.append(memory)

    def query(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """Query memories by semantic similarity"""
        if not self.embedding_model or not self.memories:
            return self.memories[:k]

        # Compute query embedding
        query_embedding = self._compute_embedding(query)

        # Compute similarities
        similarities = []
        for memory in self.memories:
            if memory.embedding is not None:
                sim = F.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    memory.embedding.unsqueeze(0)
                ).item()
                similarities.append((memory, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [mem for mem, sim in similarities[:k]]

    def _compute_embedding(self, text: str) -> torch.Tensor:
        """Compute embedding for text"""
        # In practice, use actual embedding model
        return torch.randn(768)  # Dummy embedding


class MemoryStream:
    """
    Memory stream combining short-term and long-term memory.

    Inspired by Generative Agents paper.

    Example:
        >>> memory = MemoryStream()
        >>> memory.add("User asks about weather")
        >>> memory.add("System responds with forecast")
        >>>
        >>> # Query for relevant memories
        >>> context = memory.get_relevant_context(
        ...     "What did I ask about?",
        ...     k=3
        ... )
    """

    def __init__(
        self,
        stm_capacity: int = 10,
        embedding_model: Optional[nn.Module] = None
    ):
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.ltm = LongTermMemory(embedding_model=embedding_model)

    def add(self, content: str, importance: float = 0.5):
        """Add to both short-term and long-term memory"""
        self.stm.add(content, importance)
        self.ltm.add(content, importance)

    def get_relevant_context(
        self,
        query: str,
        k: int = 5,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
        relevance_weight: float = 0.4
    ) -> str:
        """
        Get relevant context using recency, importance, and relevance.

        Combines:
        - Recency: Recent memories
        - Importance: Important memories
        - Relevance: Semantically relevant memories
        """
        # Get from long-term memory
        relevant_memories = self.ltm.query(query, k=k * 2)

        # Score by recency + importance + relevance
        # (In practice, implement proper scoring)

        # Get top-k
        top_memories = relevant_memories[:k]

        # Combine with short-term memory
        stm_context = self.stm.get_context()
        ltm_context = " ".join([m.content for m in top_memories])

        return f"{stm_context} {ltm_context}"


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Agent Systems - Tool Use, Multi-Agent, Planning, Memory")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Tool Use Agent")
    print("=" * 80)
    print("""
Example: Agent uses calculator and search tools

User: "What is the population of Tokyo times 2?"

Agent trajectory:
1. Thought: Need to find population of Tokyo
2. TOOL[search](query="population of Tokyo")
3. Result: "14 million"
4. Thought: Now need to multiply by 2
5. TOOL[calculator](expression="14000000 * 2")
6. Result: "28000000"
7. Final: "The population of Tokyo times 2 is 28 million"

This enables agents to:
- Access external information (search, APIs)
- Perform calculations (calculator, code execution)
- Take actions (send email, book appointment)
""")

    print("\n" + "=" * 80)
    print("Multi-Agent System")
    print("=" * 80)
    print("""
Example: Leader delegates to workers

Task: "Build a website"

Leader:
  Decomposes into:
  - "Design the layout"
  - "Write HTML/CSS"
  - "Deploy to server"

Worker 1: Executes "Design the layout"
Worker 2: Executes "Write HTML/CSS"
Worker 3: Executes "Deploy to server"

Leader: Aggregates results -> Complete website

Benefits:
- Parallelization: Multiple agents work simultaneously
- Specialization: Each agent can have different tools/expertise
- Robustness: If one agent fails, others continue
""")

    print("\n" + "=" * 80)
    print("Planning")
    print("=" * 80)
    print("""
Example: Hierarchical task planning

Goal: "Launch a product"

High-level plan:
1. Market research
2. Product development
3. Marketing campaign
4. Launch event

Each step can be further decomposed:

"Product development" ->
  - Design prototypes
  - User testing
  - Iterate based on feedback
  - Final production

"Marketing campaign" ->
  - Identify target audience
  - Create content
  - Run ads
  - Measure engagement

Enables:
- Complex goal achievement
- Adaptive replanning
- Progress tracking
""")

    print("\n" + "=" * 80)
    print("Memory Systems")
    print("=" * 80)
    print("""
Short-Term Memory (STM):
- Recent conversation history
- Working memory
- Limited capacity (e.g., 10 items)
- Example: "User mentioned they like pizza"

Long-Term Memory (LTM):
- Persistent storage
- Semantic search
- Unbounded capacity
- Example: "User's birthday is June 15"

Memory Stream:
- Combines STM + LTM
- Retrieves by recency, importance, relevance
- Example:
  Query: "What's my favorite food?"
  Retrieves: "User likes pizza" (high relevance)
            "Last week, user ordered pizza" (high recency)

Enables:
- Personalization: Remember user preferences
- Continuity: Maintain context across sessions
- Learning: Build up knowledge over time
""")

    print("\n" + "=" * 80)
    print("Complete Agent Architecture")
    print("=" * 80)
    print("""
Fully autonomous agent = All components combined:

Input: "Help me plan a trip to Japan"

1. Memory: Recall "User likes temples and food"
2. Planning: Create plan
   - Research destinations
   - Check flights
   - Book hotels
   - Plan itinerary
3. Tools: Use search, booking APIs
4. Multi-Agent: Delegate to specialized agents
   - Travel agent: Finds flights
   - Hotel agent: Books accommodations
   - Itinerary agent: Plans activities
5. Execute: Carry out plan with tools
6. Memory: Store trip details for future

Result: Complete trip booked and planned!

This is the foundation for:
- Personal assistants (like ChatGPT plugins)
- AutoGPT, BabyAGI
- Autonomous coding agents
- Research assistants
""")

    print("=" * 80)
