# core/interfaces.py

from abc import ABC, abstractmethod

class BrainModule(ABC):
    @abstractmethod
    def process(self, data):
        pass
