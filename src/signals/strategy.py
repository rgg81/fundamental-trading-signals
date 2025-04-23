import pandas as pd
import numpy as np
import random
from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, past_data, current_data):
        """
        Implement your strategy logic here.
        This method should return a tuple: (signal, amount)
        signal: 1 for buy, 0 for hold/sell
        amount: the amount to trade
        """
        pass

class RandomStrategy(Strategy):
    def generate_signal(self, past_data, current_data):
        """
        Generate a random signal alternating between buy and sell.
        Returns: (signal, amount)
        signal: 1 for buy, 0 for hold/sell
        amount: the amount to trade (always 1)
        """
        # Generate a random signal (0 or 1)
        signal = random.randint(0, 1)
        
        # Amount is always 1
        amount = 10
        
        return signal, amount



