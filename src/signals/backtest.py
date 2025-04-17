import pandas as pd
from .strategy import RandomStrategy

class Backtest:
    def __init__(self, strategy, max_amount=10, stop_loss=0.015, close_col='Close'):
        self.strategy = strategy
        self.max_amount = max_amount
        self.stop_loss = stop_loss
        self.close_col = close_col

    def run(self, data):
        """
        Run the backtest on the given data.
        :param data: DataFrame with columns ['Date', 'Close', 'Feature1', 'Feature2', ...]
        :return: DataFrame with backtest results
        """
        results = []
        data = data.sort_values('Date')  # Ensure data is sorted by date

        for i in range(len(data)):
            current_data = data.iloc[i]
            current_data_frame = data.iloc[i:1 + i]
            past_data = data.iloc[:i]

            if past_data.empty or i < 160:  # Ensure we have enough past data for the strategy
                # Skip the first row since there's no past data
                continue

            signal, amount = self.strategy.generate_signal(past_data, current_data_frame)
            profit_loss = 0

            if signal == 1:  # Buy signal
                if i + 1 < len(data):
                    next_close = data.iloc[i + 1][self.close_col]
                    # percentage change
                    profit_loss = ((next_close - current_data[self.close_col]) / current_data[self.close_col]) * (amount / self.max_amount)
                    profit_loss = max(min(profit_loss, self.stop_loss), -self.stop_loss)
            else:
                if i + 1 < len(data):
                    next_close = data.iloc[i + 1][self.close_col]
                    # percentage change
                    profit_loss = ((current_data[self.close_col] - next_close) / current_data[self.close_col]) * (amount / self.max_amount)
                    profit_loss = max(min(profit_loss, self.stop_loss), -self.stop_loss)
            result = {
                'Date': current_data['Date'],
                'Signal': signal,
                'Amount': amount,
                'Return': profit_loss
            }
            print(f"Date: {current_data['Date']}, Signal: {signal}, Amount: {amount}, Return: {profit_loss}")
            results.append(result)

        return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=12, freq='M'),
        'Close': [100, 105, 102, 110, 120, 115, 125, 130, 128, 135, 140, 145],
        'Feature1': range(12),
        'Feature2': range(12, 24)
    })
    
    # Test with the RandomStrategy
    random_strategy = RandomStrategy()
    random_backtest = Backtest(random_strategy)
    random_results = random_backtest.run(data)
    print("\nRandom Strategy Results:")
    print(random_results)