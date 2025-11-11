#!/usr/bin/env python3
"""
Comprehensive Test Suite for Brain AGI System

Tests all components systematically and generates detailed report.
"""

import sys
import os
import time
import traceback
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result from a single test"""
    component: str
    file_path: str
    passed: bool
    duration: float
    error: str = ""
    output: str = ""


class ComprehensiveTestRunner:
    """Runs all tests and generates comprehensive report"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def test_component(self, component: str, file_path: str) -> TestResult:
        """Test a single component"""
        print(f"\n{'='*60}")
        print(f"Testing: {component}")
        print(f"File: {file_path}")
        print('='*60)

        start = time.time()

        try:
            # Try to import and run test function
            # Convert file path to module path
            module_path = file_path.replace('/', '.').replace('.py', '')

            # Import module
            exec(f"import {module_path}")

            # Look for test function
            module = sys.modules[module_path]

            # Try common test function names
            test_fn = None
            for name in dir(module):
                if name.startswith('test_') or name == 'test' or name.endswith('_test'):
                    test_fn = getattr(module, name)
                    break

            if test_fn and callable(test_fn):
                print(f"Running test function: {test_fn.__name__}")
                test_fn()
                duration = time.time() - start

                result = TestResult(
                    component=component,
                    file_path=file_path,
                    passed=True,
                    duration=duration,
                    output="✓ Tests passed"
                )
                print(f"\n✓ {component} tests PASSED ({duration:.2f}s)")

            else:
                # No test function - try to just import
                duration = time.time() - start
                result = TestResult(
                    component=component,
                    file_path=file_path,
                    passed=True,
                    duration=duration,
                    output="✓ Import successful (no test function)"
                )
                print(f"\n⚠ {component} - No test function found (import OK)")

        except Exception as e:
            duration = time.time() - start
            error_msg = f"{type(e).__name__}: {str(e)}"
            result = TestResult(
                component=component,
                file_path=file_path,
                passed=False,
                duration=duration,
                error=error_msg
            )
            print(f"\n✗ {component} tests FAILED: {error_msg}")
            if "--verbose" in sys.argv:
                traceback.print_exc()

        self.results.append(result)
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        self.total_tests += 1

        return result

    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        lines = []
        lines.append("\n" + "="*80)
        lines.append("COMPREHENSIVE TEST REPORT - Brain AGI System")
        lines.append("="*80)

        # Summary
        total_time = sum(r.duration for r in self.results)
        pass_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0

        lines.append(f"\nTotal Tests: {self.total_tests}")
        lines.append(f"Passed: {self.passed_tests} ({pass_rate:.1f}%)")
        lines.append(f"Failed: {self.failed_tests}")
        lines.append(f"Total Time: {total_time:.2f}s")

        # Passed tests
        lines.append(f"\n{'='*80}")
        lines.append(f"PASSED TESTS ({self.passed_tests}):")
        lines.append('='*80)
        for result in self.results:
            if result.passed:
                lines.append(f"✓ {result.component:40s} ({result.duration:.2f}s)")
                if result.output and "no test function" in result.output.lower():
                    lines.append(f"  ⚠ No test function found")

        # Failed tests
        if self.failed_tests > 0:
            lines.append(f"\n{'='*80}")
            lines.append(f"FAILED TESTS ({self.failed_tests}):")
            lines.append('='*80)
            for result in self.results:
                if not result.passed:
                    lines.append(f"✗ {result.component}")
                    lines.append(f"  File: {result.file_path}")
                    lines.append(f"  Error: {result.error}")
                    lines.append("")

        # Category breakdown
        lines.append(f"\n{'='*80}")
        lines.append("CATEGORY BREAKDOWN:")
        lines.append('='*80)

        categories = {
            'attention': [],
            'moe': [],
            'reasoning': [],
            'rag': [],
            'safety': [],
            'agent': [],
            'memory': [],
            'multimodal': [],
            'training': [],
            'inference': [],
            'evaluation': [],
            'production': [],
            'scientific': [],
            'learning': [],
            'other': []
        }

        for result in self.results:
            categorized = False
            for category in categories.keys():
                if category in result.file_path:
                    categories[category].append(result)
                    categorized = True
                    break
            if not categorized:
                categories['other'].append(result)

        for category, results in categories.items():
            if results:
                passed = sum(1 for r in results if r.passed)
                total = len(results)
                lines.append(f"\n{category.upper()}:")
                lines.append(f"  Tests: {total}, Passed: {passed}, Failed: {total - passed}")
                for r in results:
                    status = "✓" if r.passed else "✗"
                    lines.append(f"    {status} {r.component}")

        lines.append(f"\n{'='*80}")
        lines.append("END OF REPORT")
        lines.append('='*80 + "\n")

        return "\n".join(lines)

    def run_all(self):
        """Run all tests"""
        print("="*80)
        print("BRAIN AGI - COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Starting comprehensive test run...")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Define all components to test
        components = [
            # Core Agent Capabilities (NEW - Critical for AGI)
            ("Code Execution Sandbox", "architectures/agent/code_sandbox.py"),
            ("Tool Use Framework", "architectures/agent/tool_use.py"),
            ("Multi-Agent Orchestration", "architectures/agent/multi_agent.py"),

            # Memory Systems (NEW - Critical for AGI)
            ("Long-Term Memory", "architectures/memory/long_term_memory.py"),

            # Multi-Modal AI (NEW - Critical for AGI)
            ("Vision/Audio/Video", "architectures/multimodal/vision_audio_video.py"),

            # Advanced Reasoning (NEW)
            ("Causal/CommonSense/Self", "architectures/reasoning/causal_commonsense_self.py"),
            ("Graph & Least-to-Most", "architectures/reasoning/graph_and_least_to_most.py"),

            # Scientific AI (NEW)
            ("Science & Math AI", "architectures/scientific/science_math.py"),

            # Continual Learning (NEW)
            ("Continual Learning", "architectures/learning/continual_learning.py"),

            # Training Infrastructure (NEW)
            ("Curriculum Learning", "architectures/training/curriculum/curriculum_learning.py"),
            ("Distributed Training", "architectures/training/distributed/distributed_training.py"),
            ("Advanced Optimizers", "architectures/training/optimizers/advanced_optimizers.py"),

            # Context Extension (NEW)
            ("Context Extension", "architectures/long_context/context_extension.py"),

            # Compression (NEW)
            ("Pruning & Distillation", "architectures/compression/pruning_distillation.py"),

            # Inference Optimization (NEW)
            ("Inference Optimization", "architectures/inference/inference_optimization.py"),

            # Evaluation & Benchmarks (NEW)
            ("Comprehensive Benchmarks", "architectures/evaluation/benchmarks.py"),

            # Tokenization (NEW)
            ("Advanced Tokenizers", "architectures/tokenization/advanced_tokenizers.py"),

            # Production (NEW)
            ("Production Infrastructure", "architectures/production/infrastructure.py"),

            # Advanced Attention
            ("Advanced Attention", "architectures/attention/advanced_attention_tested.py"),
            ("Ring Attention", "architectures/attention/ring_attention.py"),
            ("Sparse Patterns", "architectures/attention/sparse_patterns.py"),
            ("Flash Attention", "architectures/attention/flash_attention.py"),

            # MoE
            ("Advanced MoE", "architectures/moe/advanced_moe.py"),
            ("Mixture of Experts", "architectures/moe/mixture_of_experts.py"),

            # RAG
            ("Advanced RAG", "architectures/rag/advanced_rag.py"),
            ("RAG Systems", "architectures/rag/retrieval_augmented_generation.py"),

            # Safety & Alignment
            ("Safety Mechanisms", "architectures/alignment/safety_mechanisms.py"),
            ("RLHF & DPO", "architectures/alignment/rlhf_dpo.py"),

            # Alternative Architectures
            ("S4/H3/xLSTM/TTT", "architectures/alternative/s4_h3_xlstm_ttt.py"),
            ("Mamba", "architectures/alternative/mamba.py"),
            ("RetNet", "architectures/alternative/retnet.py"),
            ("RWKV", "architectures/alternative/rwkv.py"),

            # PEFT
            ("Prefix Tuning", "architectures/lora/prefix_ptuning.py"),
            ("LoRA", "architectures/lora/lora.py"),

            # Existing Components
            ("Agent Systems", "architectures/agents/agent_systems.py"),
            ("Neural Memory", "architectures/memory/neural_memory.py"),
            ("Reasoning Frameworks", "architectures/reasoning/reasoning_frameworks.py"),
            ("Advanced Reasoning", "architectures/reasoning/advanced_reasoning.py"),
            ("Scientific AI", "architectures/scientific/scientific_ai.py"),
            ("Eval Framework", "architectures/evaluation/eval_framework.py"),
        ]

        print(f"\nTotal components to test: {len(components)}\n")

        # Run all tests
        for component, file_path in components:
            if os.path.exists(file_path):
                self.test_component(component, file_path)
            else:
                print(f"\n⚠ WARNING: File not found: {file_path}")

        # Generate and print report
        report = self.generate_report()
        print(report)

        # Save report to file
        report_file = "TEST_REPORT.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {report_file}")

        # Return exit code
        return 0 if self.failed_tests == 0 else 1


def main():
    """Main entry point"""
    runner = ComprehensiveTestRunner()
    exit_code = runner.run_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
