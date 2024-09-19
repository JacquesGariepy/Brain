# plugins/plugin_interface.py

from abc import ABC, abstractmethod

class PluginInterface(ABC):
    @abstractmethod
    def process(self, data):
        pass
