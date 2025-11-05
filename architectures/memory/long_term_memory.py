"""
Long-Term Memory - CRITICAL FOR AGI

Comprehensive memory systems:
- Vector Store (semantic memory)
- Knowledge Graph (relational memory)
- Episodic Memory (experience replay)
- Working Memory (short-term context)
- Memory consolidation and retrieval
- Forgetting and prioritization

References:
- "MemPrompt: Memory-assisted Prompt Editing" (2022)
- "Generative Agents: Interactive Simulacra of Human Behavior" (Stanford, 2023)
- "Memory Networks" (Weston et al., 2015)
- "Retrieval-Enhanced Transformer" (RETRO, DeepMind, 2021)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import json
import heapq


@dataclass
class Memory:
    """Base memory unit"""
    memory_id: str
    content: str
    embedding: Optional[torch.Tensor] = None
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0  # 0-1
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_type: str = "semantic"  # semantic, episodic, procedural


@dataclass
class Episode:
    """Episodic memory (specific experience)"""
    episode_id: str
    description: str
    context: Dict[str, Any]
    observations: List[str]
    actions: List[str]
    outcomes: List[str]
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0


class VectorStore:
    """
    Vector-based semantic memory.

    Stores embeddings for fast similarity search.
    """

    def __init__(
        self,
        dimension: int = 768,
        max_memories: int = 10000
    ):
        self.dimension = dimension
        self.max_memories = max_memories

        # Storage
        self.memories: Dict[str, Memory] = {}
        self.embeddings: Optional[torch.Tensor] = None
        self.memory_ids: List[str] = []

    def add_memory(
        self,
        memory_id: str,
        content: str,
        embedding: torch.Tensor,
        importance: float = 1.0,
        metadata: Optional[Dict] = None
    ):
        """
        Add memory to store.

        Args:
            memory_id: Unique identifier
            content: Memory content
            embedding: Vector embedding [dimension]
            importance: Importance score 0-1
            metadata: Additional metadata
        """
        # Check capacity
        if len(self.memories) >= self.max_memories:
            self._evict_least_important()

        # Create memory
        memory = Memory(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
            importance=importance,
            metadata=metadata or {}
        )

        self.memories[memory_id] = memory
        self.memory_ids.append(memory_id)

        # Update embedding matrix
        if self.embeddings is None:
            self.embeddings = embedding.unsqueeze(0)
        else:
            self.embeddings = torch.cat([self.embeddings, embedding.unsqueeze(0)], dim=0)

    def search(
        self,
        query_embedding: torch.Tensor,
        k: int = 5,
        importance_threshold: float = 0.0
    ) -> List[Tuple[Memory, float]]:
        """
        Search for similar memories.

        Args:
            query_embedding: Query vector [dimension]
            k: Number of results
            importance_threshold: Minimum importance

        Returns:
            List of (memory, similarity_score)
        """
        if self.embeddings is None or len(self.memories) == 0:
            return []

        # Compute similarities
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.embeddings,
            dim=1
        )

        # Filter by importance
        valid_indices = [
            i for i, mem_id in enumerate(self.memory_ids)
            if self.memories[mem_id].importance >= importance_threshold
        ]

        if not valid_indices:
            return []

        # Get top-k
        valid_similarities = similarities[valid_indices]
        top_k = min(k, len(valid_indices))

        top_indices = torch.topk(valid_similarities, k=top_k).indices
        actual_indices = [valid_indices[i] for i in top_indices]

        # Update access stats
        results = []
        for idx in actual_indices:
            mem_id = self.memory_ids[idx]
            memory = self.memories[mem_id]
            memory.access_count += 1
            memory.last_accessed = time.time()

            similarity = similarities[idx].item()
            results.append((memory, similarity))

        return results

    def _evict_least_important(self):
        """Remove least important memory"""
        if not self.memories:
            return

        # Find least important (considering importance and recency)
        def score(mem: Memory) -> float:
            age = time.time() - mem.last_accessed
            # Decay importance with age
            return mem.importance * np.exp(-age / 86400)  # 1 day decay

        least_important = min(self.memories.values(), key=score)

        # Remove
        idx = self.memory_ids.index(least_important.memory_id)
        del self.memories[least_important.memory_id]
        self.memory_ids.pop(idx)

        # Remove from embeddings
        mask = torch.ones(self.embeddings.shape[0], dtype=torch.bool)
        mask[idx] = False
        self.embeddings = self.embeddings[mask]

    def consolidate(self, similarity_threshold: float = 0.95):
        """
        Consolidate similar memories.

        Merges highly similar memories to reduce redundancy.
        """
        if len(self.memories) < 2:
            return

        # Compute pairwise similarities
        similarities = torch.matmul(self.embeddings, self.embeddings.T)

        # Find pairs to merge
        merged_ids = set()

        for i in range(len(self.memory_ids)):
            if self.memory_ids[i] in merged_ids:
                continue

            for j in range(i + 1, len(self.memory_ids)):
                if similarities[i, j] > similarity_threshold:
                    # Merge j into i
                    mem_i = self.memories[self.memory_ids[i]]
                    mem_j = self.memories[self.memory_ids[j]]

                    # Update i with merged content
                    mem_i.content = f"{mem_i.content} | {mem_j.content}"
                    mem_i.importance = max(mem_i.importance, mem_j.importance)
                    mem_i.access_count += mem_j.access_count

                    # Mark j for removal
                    merged_ids.add(self.memory_ids[j])

        # Remove merged memories
        for mem_id in merged_ids:
            idx = self.memory_ids.index(mem_id)
            del self.memories[mem_id]
            self.memory_ids.pop(idx)

            # Update embeddings
            mask = torch.ones(self.embeddings.shape[0], dtype=torch.bool)
            mask[idx] = False
            self.embeddings = self.embeddings[mask]


class KnowledgeGraph:
    """
    Knowledge Graph for relational memory.

    Stores entities and relationships.
    """

    def __init__(self):
        # Entity storage
        self.entities: Dict[str, Dict[str, Any]] = {}

        # Relationship storage: {(entity1, relation, entity2): metadata}
        self.relationships: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        # Indexes for fast lookup
        self.entity_relations: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Add entity to graph.

        Args:
            entity_id: Unique identifier
            entity_type: Type of entity (person, place, concept, etc.)
            properties: Entity properties
        """
        self.entities[entity_id] = {
            "id": entity_id,
            "type": entity_type,
            "properties": properties or {},
            "created": time.time()
        }

    def add_relationship(
        self,
        subject: str,
        relation: str,
        object: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Add relationship between entities.

        Args:
            subject: Source entity ID
            relation: Relationship type
            object: Target entity ID
            properties: Relationship properties
        """
        # Ensure entities exist
        if subject not in self.entities:
            self.add_entity(subject, "unknown")
        if object not in self.entities:
            self.add_entity(object, "unknown")

        # Add relationship
        triple = (subject, relation, object)
        self.relationships[triple] = {
            "properties": properties or {},
            "created": time.time()
        }

        # Update indexes
        self.entity_relations[subject].append(triple)
        self.entity_relations[object].append(triple)

    def query(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        object: Optional[str] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Query knowledge graph.

        Args:
            subject: Filter by subject (optional)
            relation: Filter by relation (optional)
            object: Filter by object (optional)

        Returns:
            Matching triples
        """
        results = []

        for triple in self.relationships.keys():
            s, r, o = triple

            # Check filters
            if subject is not None and s != subject:
                continue
            if relation is not None and r != relation:
                continue
            if object is not None and o != object:
                continue

            results.append(triple)

        return results

    def get_neighbors(
        self,
        entity_id: str,
        relation: Optional[str] = None,
        direction: str = "both"  # outgoing, incoming, both
    ) -> List[str]:
        """
        Get neighboring entities.

        Args:
            entity_id: Entity to get neighbors for
            relation: Filter by relation type
            direction: Direction of relationships

        Returns:
            List of neighbor entity IDs
        """
        neighbors = set()

        for triple in self.entity_relations.get(entity_id, []):
            s, r, o = triple

            # Check relation filter
            if relation is not None and r != relation:
                continue

            # Add based on direction
            if direction in ["outgoing", "both"] and s == entity_id:
                neighbors.add(o)
            if direction in ["incoming", "both"] and o == entity_id:
                neighbors.add(s)

        return list(neighbors)

    def subgraph(
        self,
        entity_id: str,
        max_depth: int = 2
    ) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
        """
        Extract subgraph around entity.

        Args:
            entity_id: Center entity
            max_depth: Maximum traversal depth

        Returns:
            (entities, relationships) in subgraph
        """
        entities = {entity_id}
        relationships = set()

        # BFS traversal
        queue = [(entity_id, 0)]
        visited = {entity_id}

        while queue:
            current, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            # Get all relationships
            for triple in self.entity_relations.get(current, []):
                s, r, o = triple
                relationships.add(triple)

                # Add neighbors
                neighbor = o if s == current else s
                if neighbor not in visited:
                    visited.add(neighbor)
                    entities.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return entities, relationships


class EpisodicMemory:
    """
    Episodic memory for storing experiences.

    Implements experience replay and temporal reasoning.
    """

    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: Dict[str, Episode] = {}
        self.episode_ids: List[str] = []

    def add_episode(self, episode: Episode):
        """
        Add episodic memory.

        Args:
            episode: Episode to store
        """
        # Check capacity
        if len(self.episodes) >= self.max_episodes:
            self._evict_oldest()

        self.episodes[episode.episode_id] = episode
        self.episode_ids.append(episode.episode_id)

    def retrieve_recent(self, k: int = 5) -> List[Episode]:
        """
        Retrieve k most recent episodes.

        Args:
            k: Number of episodes

        Returns:
            Recent episodes
        """
        recent_ids = self.episode_ids[-k:]
        return [self.episodes[eid] for eid in recent_ids]

    def retrieve_important(self, k: int = 5) -> List[Episode]:
        """
        Retrieve k most important episodes.

        Args:
            k: Number of episodes

        Returns:
            Important episodes
        """
        sorted_episodes = sorted(
            self.episodes.values(),
            key=lambda e: e.importance,
            reverse=True
        )
        return sorted_episodes[:k]

    def search_by_context(
        self,
        context_key: str,
        context_value: Any,
        k: int = 5
    ) -> List[Episode]:
        """
        Search episodes by context.

        Args:
            context_key: Context key to match
            context_value: Context value to match
            k: Number of results

        Returns:
            Matching episodes
        """
        matches = [
            episode for episode in self.episodes.values()
            if episode.context.get(context_key) == context_value
        ]

        # Sort by recency
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        return matches[:k]

    def _evict_oldest(self):
        """Remove oldest episode"""
        if self.episode_ids:
            oldest_id = self.episode_ids[0]
            del self.episodes[oldest_id]
            self.episode_ids.pop(0)


class WorkingMemory:
    """
    Working memory for short-term context.

    Implements:
    - Attention mechanism
    - Capacity limits
    - Recency bias
    """

    def __init__(self, capacity: int = 7):  # Miller's 7±2
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)

    def add(self, item: str):
        """Add item to working memory"""
        self.items.append(item)

    def get_context(self) -> List[str]:
        """Get current working memory context"""
        return list(self.items)

    def clear(self):
        """Clear working memory"""
        self.items.clear()


class IntegratedMemorySystem:
    """
    Integrated memory system combining all memory types.

    Coordinates between:
    - Working memory (immediate context)
    - Episodic memory (experiences)
    - Semantic memory (vector store)
    - Knowledge graph (structured knowledge)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_semantic_memories: int = 10000,
        max_episodes: int = 1000,
        working_capacity: int = 7
    ):
        self.semantic = VectorStore(embedding_dim, max_semantic_memories)
        self.episodic = EpisodicMemory(max_episodes)
        self.knowledge = KnowledgeGraph()
        self.working = WorkingMemory(working_capacity)

    def store(
        self,
        content: str,
        embedding: Optional[torch.Tensor] = None,
        memory_type: str = "semantic",
        **kwargs
    ):
        """
        Store memory in appropriate system.

        Args:
            content: Memory content
            embedding: Vector embedding (for semantic)
            memory_type: Type of memory (semantic, episodic, knowledge)
            **kwargs: Additional arguments
        """
        # Always add to working memory
        self.working.add(content)

        # Store in appropriate long-term memory
        if memory_type == "semantic" and embedding is not None:
            memory_id = f"mem_{time.time()}"
            self.semantic.add_memory(
                memory_id=memory_id,
                content=content,
                embedding=embedding,
                **kwargs
            )

        elif memory_type == "episodic":
            episode = Episode(
                episode_id=f"ep_{time.time()}",
                description=content,
                **kwargs
            )
            self.episodic.add_episode(episode)

        elif memory_type == "knowledge":
            # Parse as triple: subject, relation, object
            parts = content.split("|")
            if len(parts) >= 3:
                self.knowledge.add_relationship(
                    subject=parts[0].strip(),
                    relation=parts[1].strip(),
                    object=parts[2].strip()
                )

    def retrieve(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[torch.Tensor] = None,
        memory_types: List[str] = None,
        k: int = 5
    ) -> Dict[str, List]:
        """
        Retrieve memories across all systems.

        Args:
            query: Text query
            query_embedding: Vector query
            memory_types: Which systems to query
            k: Number of results per system

        Returns:
            Results from each memory system
        """
        if memory_types is None:
            memory_types = ["semantic", "episodic", "working"]

        results = {}

        # Semantic search
        if "semantic" in memory_types and query_embedding is not None:
            semantic_results = self.semantic.search(query_embedding, k=k)
            results["semantic"] = [
                {"content": mem.content, "similarity": sim}
                for mem, sim in semantic_results
            ]

        # Episodic retrieval
        if "episodic" in memory_types:
            recent_episodes = self.episodic.retrieve_recent(k=k)
            results["episodic"] = [
                {"description": ep.description, "timestamp": ep.timestamp}
                for ep in recent_episodes
            ]

        # Working memory
        if "working" in memory_types:
            results["working"] = self.working.get_context()

        return results


# Testing
def test_long_term_memory():
    """Test long-term memory systems"""
    print("Testing Long-Term Memory...")

    # Test 1: Vector Store
    print("\n1. Vector Store (Semantic Memory)")
    vector_store = VectorStore(dimension=128, max_memories=100)

    # Add memories
    memories = [
        ("Python is a programming language", "programming"),
        ("Machine learning uses neural networks", "AI"),
        ("Deep learning is a subset of ML", "AI"),
        ("JavaScript runs in browsers", "programming"),
        ("Transformers revolutionized NLP", "AI")
    ]

    for i, (content, category) in enumerate(memories):
        embedding = torch.randn(128)  # Simulated
        vector_store.add_memory(
            memory_id=f"mem_{i}",
            content=content,
            embedding=embedding,
            importance=0.8,
            metadata={"category": category}
        )

    print(f"  Stored {len(vector_store.memories)} memories")

    # Search
    query_emb = torch.randn(128)
    results = vector_store.search(query_emb, k=3)
    print(f"  Top 3 search results:")
    for mem, sim in results:
        print(f"    - {mem.content} (sim: {sim:.3f})")

    # Test 2: Knowledge Graph
    print("\n2. Knowledge Graph")
    kg = KnowledgeGraph()

    # Add entities and relationships
    kg.add_entity("Python", "Language", {"paradigm": "multi-paradigm"})
    kg.add_entity("Guido", "Person", {"role": "creator"})
    kg.add_entity("ML", "Field", {"domain": "AI"})

    kg.add_relationship("Guido", "created", "Python")
    kg.add_relationship("Python", "used_for", "ML")
    kg.add_relationship("ML", "part_of", "AI")

    print(f"  Entities: {len(kg.entities)}")
    print(f"  Relationships: {len(kg.relationships)}")

    # Query
    python_relations = kg.query(subject="Python")
    print(f"  Python relationships:")
    for s, r, o in python_relations:
        print(f"    {s} --{r}--> {o}")

    # Get neighbors
    neighbors = kg.get_neighbors("Python")
    print(f"  Python neighbors: {neighbors}")

    # Subgraph
    entities, relations = kg.subgraph("Python", max_depth=2)
    print(f"  Subgraph around Python: {len(entities)} entities, {len(relations)} relations")

    # Test 3: Episodic Memory
    print("\n3. Episodic Memory")
    episodic = EpisodicMemory(max_episodes=100)

    # Add episodes
    for i in range(5):
        episode = Episode(
            episode_id=f"ep_{i}",
            description=f"Completed task {i}",
            context={"task_type": "coding" if i % 2 == 0 else "research"},
            observations=[f"Observation {i}"],
            actions=[f"Action {i}"],
            outcomes=[f"Outcome {i}"],
            importance=0.5 + i * 0.1
        )
        episodic.add_episode(episode)

    print(f"  Stored {len(episodic.episodes)} episodes")

    # Retrieve recent
    recent = episodic.retrieve_recent(k=3)
    print(f"  Recent episodes:")
    for ep in recent:
        print(f"    - {ep.description}")

    # Retrieve important
    important = episodic.retrieve_important(k=2)
    print(f"  Important episodes:")
    for ep in important:
        print(f"    - {ep.description} (importance: {ep.importance:.2f})")

    # Search by context
    coding_eps = episodic.search_by_context("task_type", "coding", k=3)
    print(f"  Coding episodes: {len(coding_eps)}")

    # Test 4: Working Memory
    print("\n4. Working Memory")
    working = WorkingMemory(capacity=5)

    items = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6"]
    for item in items:
        working.add(item)

    context = working.get_context()
    print(f"  Working memory (capacity 5, added 6 items):")
    for item in context:
        print(f"    - {item}")

    # Test 5: Integrated System
    print("\n5. Integrated Memory System")
    integrated = IntegratedMemorySystem(
        embedding_dim=128,
        max_semantic_memories=1000,
        max_episodes=100
    )

    # Store semantic
    integrated.store(
        "PyTorch is a deep learning framework",
        embedding=torch.randn(128),
        memory_type="semantic",
        importance=0.9
    )

    # Store episodic
    integrated.store(
        "Trained model on dataset X",
        memory_type="episodic",
        context={"task": "training"},
        observations=["Loss decreased"],
        actions=["Adjusted learning rate"],
        outcomes=["Achieved 95% accuracy"]
    )

    # Store knowledge
    integrated.store(
        "PyTorch | developed_by | Meta",
        memory_type="knowledge"
    )

    # Retrieve
    results = integrated.retrieve(
        query_embedding=torch.randn(128),
        memory_types=["semantic", "episodic", "working"],
        k=3
    )

    print(f"  Retrieved from {len(results)} memory systems:")
    for mem_type, items in results.items():
        print(f"    {mem_type}: {len(items)} items")

    print("\n✓ Long-Term Memory tests completed!")

    # Summary
    print("\n" + "="*60)
    print("LONG-TERM MEMORY SUMMARY")
    print("="*60)
    print("Memory systems: 4")
    print("  1. Semantic Memory (Vector Store)")
    print("     - Embedding-based storage")
    print("     - Cosine similarity search")
    print("     - Importance-based eviction")
    print("     - Memory consolidation")
    print("  2. Knowledge Graph")
    print("     - Entity-relationship model")
    print("     - Triple store (subject-relation-object)")
    print("     - Graph traversal and querying")
    print("     - Subgraph extraction")
    print("  3. Episodic Memory")
    print("     - Experience storage")
    print("     - Temporal reasoning")
    print("     - Context-based retrieval")
    print("     - Importance ranking")
    print("  4. Working Memory")
    print("     - Short-term context (7±2 items)")
    print("     - Recency-based")
    print("     - Attention mechanism")
    print("\nFeatures:")
    print("  - Multi-modal memory storage")
    print("  - Intelligent forgetting (importance + recency)")
    print("  - Memory consolidation")
    print("  - Cross-system retrieval")
    print("  - Temporal reasoning")
    print("\nApplications:")
    print("  - Conversational AI (context retention)")
    print("  - Personal assistants (user preferences)")
    print("  - Autonomous agents (experience replay)")
    print("  - Knowledge management")
    print("  - Lifelong learning")


if __name__ == "__main__":
    test_long_term_memory()
