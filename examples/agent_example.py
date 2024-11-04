from core.brain import Brain

class Agent:
    def __init__(self):
        self.brain = Brain()

    def process_sensory_input(self, sensory_input):
        return self.brain.process(sensory_input)

    def make_decision(self, data):
        return self.brain.modules['reasoning'].process(data)

    def learn(self, text):
        self.brain.inject_knowledge(text)

    def communicate(self, prompt):
        return self.brain.modules['language'].generate_sentence(prompt)

if __name__ == "__main__":
    agent = Agent()

    # Process sensory input
    sensory_input = "raw sensory data"
    processed_data = agent.process_sensory_input(sensory_input)
    print(f"Processed Data: {processed_data}")

    # Make a decision
    decision = agent.make_decision(processed_data)
    print(f"Decision: {decision}")

    # Learn new knowledge
    knowledge_text = "Artificial intelligence is a rapidly growing field."
    agent.learn(knowledge_text)

    # Communicate
    prompt = "Tell me about AI."
    response = agent.communicate(prompt)
    print(f"Response: {response}")
