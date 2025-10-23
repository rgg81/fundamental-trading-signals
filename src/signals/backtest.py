import pandas as pd
from signals.strategy import RandomStrategy

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
        step_size = 6  # You can adjust the step size if needed

        for i in range(0, len(data), step_size):
            current_step = i + step_size
            current_data_frame = data.iloc[i:current_step]
            past_data = data.iloc[:current_step]

            if past_data.empty or i < 154 or current_data_frame.empty:  # Ensure we have enough past data for the strategy
                # Skip the first row since there's no past data
                continue

            #if current_data_frame.iloc[0]['Date'].year != 2020:
            #    continue

            signals, amounts = self.strategy.generate_signal(past_data, current_data_frame)
            if signals is None: continue
            profit_loss = 0
            index_next = i + 1
            for signal, amount in zip(signals, amounts):
                current_data = data.iloc[index_next - 1]
                if signal == 1:  # Buy signal
                    if index_next < len(data):
                        next_close = data.iloc[index_next][self.close_col]
                        # percentage change
                        profit_loss = ((next_close - current_data[self.close_col]) / current_data[self.close_col]) 
                        profit_loss = max(min(profit_loss, self.stop_loss), -self.stop_loss) * (amount / self.max_amount)
                else:
                    if index_next < len(data):
                        next_close = data.iloc[index_next][self.close_col]
                        # percentage change
                        profit_loss = ((current_data[self.close_col] - next_close) / current_data[self.close_col]) 
                        profit_loss = max(min(profit_loss, self.stop_loss), -self.stop_loss) * (amount / self.max_amount)
                result = {
                    'Date': current_data['Date'],
                    'Signal': signal,
                    'Amount': amount,
                    'Return': profit_loss
                }
                print(f"*** Date: {current_data['Date']}, Label: {current_data['Label']} Signal: {signal}, Amount: {amount}, Return: {profit_loss}, current close: {current_data[self.close_col]} next close: {next_close if index_next < len(data) else 'N/A'} ***", flush=True)
                results.append(result)
                index_next += 1

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