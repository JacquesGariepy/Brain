#!/usr/bin/env python3
"""
Unit Tests for All AGI Components (No External Dependencies)

Tests logic and functionality without requiring torch or other heavy libraries.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCodeSandbox(unittest.TestCase):
    """Tests for code execution sandbox"""

    def test_python_validation(self):
        """Test Python code validation"""
        from architectures.agent.code_sandbox import PythonSandbox, SandboxConfig

        config = SandboxConfig(allowed_imports=['math'])
        sandbox = PythonSandbox(config)

        # Test safe code
        is_safe, error = sandbox.validate_code("x = 1 + 1")
        self.assertTrue(is_safe)
        self.assertIsNone(error)

        # Test dangerous code
        is_safe, error = sandbox.validate_code("eval('1+1')")
        self.assertFalse(is_safe)
        self.assertIn("eval", error)

    def test_bash_validation(self):
        """Test Bash command validation"""
        from architectures.agent.code_sandbox import BashSandbox, SandboxConfig

        config = SandboxConfig()
        sandbox = BashSandbox(config)

        # Test safe command
        is_safe, error = sandbox.validate_command("echo hello")
        self.assertTrue(is_safe)

        # Test dangerous command
        is_safe, error = sandbox.validate_command("rm -rf /")
        self.assertFalse(is_safe)

    def test_language_detection(self):
        """Test language auto-detection"""
        from architectures.agent.code_sandbox import UnifiedCodeSandbox

        sandbox = UnifiedCodeSandbox()

        self.assertEqual(sandbox.detect_language("def hello(): pass"), "python")
        self.assertEqual(sandbox.detect_language("function hello() {}"), "javascript")
        self.assertEqual(sandbox.detect_language("echo hello"), "bash")


class TestToolUse(unittest.TestCase):
    """Tests for tool use framework"""

    def test_calculator(self):
        """Test calculator tool"""
        from architectures.agent.tool_use import CalculatorTool

        calc = CalculatorTool()

        result = calc.calculate("2 + 2")
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], 4)

        result = calc.calculate("sqrt(144)")
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], 12.0)

    def test_tool_registry(self):
        """Test tool registry"""
        from architectures.agent.tool_use import ToolRegistry, ToolSpec, ToolType

        registry = ToolRegistry()

        # Register a simple tool
        def add(a, b):
            return a + b

        spec = ToolSpec(
            name="add",
            description="Add two numbers",
            parameters={"a": "int", "b": "int"},
            returns={"type": "int"},
            examples=[],
            tool_type=ToolType.CALCULATOR
        )

        registry.register("add", add, spec)

        self.assertEqual(len(registry.tools), 1)
        self.assertIn("add", registry.tools)

    def test_browser_search(self):
        """Test browser search"""
        from architectures.agent.tool_use import BrowserTool

        browser = BrowserTool()
        result = browser.search("test query", num_results=3)

        self.assertTrue(result.success)
        self.assertEqual(len(result.data['results']), 3)


class TestBenchmarks(unittest.TestCase):
    """Tests for evaluation benchmarks"""

    def test_humaneval_structure(self):
        """Test HumanEval benchmark structure"""
        from architectures.evaluation.benchmarks import HumanEvalBenchmark

        bench = HumanEvalBenchmark()
        self.assertGreater(len(bench.problems), 0)

        problem = bench.problems[0]
        self.assertIn('task_id', problem)
        self.assertIn('prompt', problem)
        self.assertIn('test', problem)

    def test_mtbench_categories(self):
        """Test MT-Bench categories"""
        from architectures.evaluation.benchmarks import MTBenchBenchmark

        bench = MTBenchBenchmark()
        self.assertEqual(len(bench.CATEGORIES), 8)
        self.assertIn('writing', bench.CATEGORIES)
        self.assertIn('coding', bench.CATEGORIES)

    def test_math_benchmark(self):
        """Test MATH benchmark"""
        from architectures.evaluation.benchmarks import MATHBenchmark

        bench = MATHBenchmark()
        self.assertEqual(bench.DIFFICULTY_LEVELS, [1, 2, 3, 4, 5])
        self.assertGreater(len(bench.problems), 0)

    def test_evaluator(self):
        """Test comprehensive evaluator"""
        from architectures.evaluation.benchmarks import ComprehensiveEvaluator

        evaluator = ComprehensiveEvaluator()
        self.assertIn("HumanEval", evaluator.benchmarks)
        self.assertIn("MT-Bench", evaluator.benchmarks)
        self.assertIn("MATH", evaluator.benchmarks)
        self.assertIn("AgentBench", evaluator.benchmarks)


class TestReasoning(unittest.TestCase):
    """Tests for reasoning systems (without torch)"""

    def test_causal_graph(self):
        """Test causal graph structure"""
        try:
            from architectures.reasoning.causal_commonsense_self import CausalGraph

            graph = CausalGraph()
            graph.add_edge("Rain", "WetGround")
            graph.add_edge("WetGround", "Slippery")

            self.assertIn("Rain", graph.nodes)
            self.assertIn("WetGround", graph.get_children("Rain"))
            self.assertIn("Rain", graph.get_parents("WetGround"))

            ancestors = graph.get_ancestors("Slippery")
            self.assertIn("WetGround", ancestors)
            self.assertIn("Rain", ancestors)
        except ImportError:
            self.skipTest("Module requires torch")

    def test_common_sense_physical(self):
        """Test physical reasoning"""
        try:
            from architectures.reasoning.causal_commonsense_self import PhysicalReasoning

            physical = PhysicalReasoning()
            plausible, reason = physical.check_physical_plausibility("A ball falls down")
            self.assertTrue(plausible)

            plausible, reason = physical.check_physical_plausibility("A ball floats up without support")
            self.assertFalse(plausible)
        except ImportError:
            self.skipTest("Module requires torch")

    def test_self_critic(self):
        """Test self-critique system"""
        try:
            from architectures.reasoning.causal_commonsense_self import SelfCritic

            critic = SelfCritic()
            critique = critic.critique("Short answer", "Explain the universe")

            self.assertIn('scores', critique)
            self.assertIn('correctness', critique['scores'])
            self.assertIn('completeness', critique['scores'])
        except ImportError:
            self.skipTest("Module requires torch")


class TestProduction(unittest.TestCase):
    """Tests for production infrastructure (without torch)"""

    def test_request_batcher(self):
        """Test request batching"""
        try:
            from architectures.production.infrastructure import RequestBatcher, Request

            batcher = RequestBatcher(max_batch_size=3, max_wait_ms=100.0)

            req1 = Request("1", "/test", {})
            req2 = Request("2", "/test", {})
            req3 = Request("3", "/test", {})

            batch = batcher.add_request(req1)
            self.assertIsNone(batch)  # Not full yet

            batch = batcher.add_request(req2)
            self.assertIsNone(batch)  # Still not full

            batch = batcher.add_request(req3)
            self.assertIsNotNone(batch)  # Now full
            self.assertEqual(len(batch), 3)
        except ImportError:
            self.skipTest("Module requires torch")

    def test_response_cache(self):
        """Test response caching"""
        try:
            from architectures.production.infrastructure import ResponseCache, Request, Response

            cache = ResponseCache(max_size=10, ttl_seconds=60.0)

            req = Request("1", "/test", {"query": "hello"})
            resp = Response("1", "world")

            # Cache miss
            cached = cache.get(req)
            self.assertIsNone(cached)

            # Cache put
            cache.put(req, resp)

            # Cache hit
            cached = cache.get(req)
            self.assertIsNotNone(cached)
            self.assertEqual(cached.data, "world")
        except ImportError:
            self.skipTest("Module requires torch")

    def test_metrics_collector(self):
        """Test metrics collection"""
        try:
            from architectures.production.infrastructure import MetricsCollector

            metrics = MetricsCollector(window_size=100)

            # Record some requests
            for i in range(10):
                metrics.record_request("/test", latency=0.1, is_error=(i % 5 == 0))

            current = metrics.get_metrics()

            self.assertEqual(current['total_requests'], 10)
            self.assertEqual(current['total_errors'], 2)
            self.assertAlmostEqual(current['error_rate'], 0.2)
        except ImportError:
            self.skipTest("Module requires torch")


class TestTokenization(unittest.TestCase):
    """Tests for tokenizers (without torch)"""

    def test_wordpiece_basic(self):
        """Test WordPiece tokenizer basic functionality"""
        try:
            from architectures.tokenization.advanced_tokenizers import WordPieceTokenizer

            tokenizer = WordPieceTokenizer()

            # Check special tokens
            self.assertIn("[PAD]", tokenizer.vocab)
            self.assertIn("[UNK]", tokenizer.vocab)
            self.assertIn("[CLS]", tokenizer.vocab)
            self.assertIn("[SEP]", tokenizer.vocab)
            self.assertIn("[MASK]", tokenizer.vocab)
        except ImportError:
            self.skipTest("Module requires imports")


def run_tests():
    """Run all unit tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCodeSandbox))
    suite.addTests(loader.loadTestsFromTestCase(TestToolUse))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarks))
    suite.addTests(loader.loadTestsFromTestCase(TestReasoning))
    suite.addTests(loader.loadTestsFromTestCase(TestProduction))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenization))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("UNIT TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
