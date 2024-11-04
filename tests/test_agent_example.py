import unittest
from examples.agent_example import Agent

class TestAgentExample(unittest.TestCase):
    def setUp(self):
        """Initialise an instance of the Agent for testing."""
        self.agent = Agent()

    def test_agent_initialization(self):
        """Test the initialization of the Brain instance."""
        self.assertIsNotNone(self.agent.brain)

    def test_process_sensory_input(self):
        """Test the processing of sensory input."""
        sensory_input = "raw sensory data"
        processed_data = self.agent.process_sensory_input(sensory_input)
        self.assertEqual(processed_data, "données_perçues")

    def test_make_decision(self):
        """Test the decision-making process."""
        data = "données_perçues"
        decision = self.agent.make_decision(data)
        self.assertEqual(decision, "données_raisonnées")

    def test_learn(self):
        """Test the learning process by injecting knowledge."""
        knowledge_text = "Artificial intelligence is a rapidly growing field."
        self.agent.learn(knowledge_text)
        self.assertIn("Artificial intelligence", self.agent.brain.modules['language'].vocabulary)

    def test_communicate(self):
        """Test the communication process."""
        prompt = "Tell me about AI."
        response = self.agent.communicate(prompt)
        self.assertTrue(len(response) > 0)

if __name__ == "__main__":
    unittest.main()
