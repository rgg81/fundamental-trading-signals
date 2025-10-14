import pandas as pd
import numpy as np
import quantstats as qs
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')

class EconomicIndicatorsFeatureEngineering:
    """
    Feature engineering class that creates economic indicators based on binary predictions
    (        print("=== Starting Economic Indicators Feature Engineering Pipeline (Binary Prediction) ===\n")for buy, -1 for sell) using a rolling time window approach.
    """
    
    def __init__(self, time_window: int = 12):
        """
        Initialize the feature engineering class.
        
        Parameters:
        -----------
        time_window : int
            Time window in months for rolling calculations (default: 12 months)
        """
        self.time_window = time_window
        
    def load_eurusd_data(self, file_path: str = "EURUSD.csv") -> pd.DataFrame:
        """
        Load EURUSD data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the EURUSD CSV file
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with Date and EURUSD_Close columns
        """
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            return df[['EURUSD_Close']]
        except Exception as e:
            raise ValueError(f"Error loading EURUSD data: {e}")
    
    def generate_monthly_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate monthly returns from daily EURUSD close prices.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with EURUSD_Close column and Date index
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with monthly returns
        """
        # Resample to monthly data (last day of month)
        monthly_prices = df.resample('ME').last()
        
        # Calculate monthly returns
        monthly_returns = monthly_prices.pct_change().dropna()
        monthly_returns.columns = ['Monthly_Return']
        
        return monthly_returns
    
    def run_feature_engineering(self, 
                               eurusd_file_path: str = "EURUSD.csv",
                               save_features: bool = True,
                               output_file: str = "economic_indicators_features.csv") -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Parameters:
        -----------
        eurusd_file_path : str
            Path to EURUSD CSV file
        save_features : bool
            Whether to save features to CSV
        output_file : str
            Output CSV filename
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with economic indicator features
        """
        print("=== Economic Indicators Feature Engineering Pipeline ===")
        
        # Step 1: Load EURUSD data
        print("1. Loading EURUSD data...")
        eurusd_data = self.load_eurusd_data(eurusd_file_path)
        print(f"   Loaded {len(eurusd_data)} daily observations")
        
        # Step 2: Generate monthly returns
        print("2. Generating monthly returns...")
        monthly_returns = self.generate_monthly_returns(eurusd_data)
        print(f"   Generated {len(monthly_returns)} monthly observations")
        
        # Step 3: Generate rolling economic indicator features
        print("3. Generating rolling economic indicator features...")
        economic_features = self.generate_rolling_features(monthly_returns)
        print(f"   Generated {economic_features.shape[1]} economic indicator features")
        print(f"   Features available for {len(economic_features)} time periods")
        
        # Step 4: Save features if requested
        if save_features:
            print(f"4. Saving features to {output_file}...")
            economic_features.to_csv(output_file)
            print(f"   Features saved successfully!")
        
        print("\n=== Feature Engineering Complete ===")
        print(f"Final dataset shape: {economic_features.shape}")
        print(f"Features generated: {list(economic_features.columns)}")
        
        return economic_features
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics using QuantStats.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of returns
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of performance metrics
        """
        if len(returns) == 0 or returns.isna().all():
            return {metric: np.nan for metric in self._get_metric_names()}
        
        # Remove NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 2:
            return {metric: np.nan for metric in self._get_metric_names()}
        
        try:
            metrics = {}
            
            # Core QuantStats metrics that definitely exist
            metrics['Cumulative_Return'] = qs.stats.compsum(clean_returns).iloc[-1] if len(clean_returns) > 0 else np.nan
            metrics['Sharpe'] = qs.stats.sharpe(clean_returns, rf=0.0)
            metrics['Smart_Sharpe'] = qs.stats.smart_sharpe(clean_returns)
            metrics['Sortino'] = qs.stats.sortino(clean_returns, rf=0.0)
            metrics['Smart_Sortino'] = qs.stats.smart_sortino(clean_returns)
            metrics['Max_Drawdown'] = qs.stats.max_drawdown(clean_returns)
            metrics['Volatility_Ann'] = qs.stats.volatility(clean_returns, annualize=True)
            metrics['Calmar'] = qs.stats.calmar(clean_returns)
            metrics['Skew'] = qs.stats.skew(clean_returns)
            metrics['Kurtosis'] = qs.stats.kurtosis(clean_returns)
            metrics['Payoff_Ratio'] = qs.stats.payoff_ratio(clean_returns)
            metrics['Profit_Factor'] = qs.stats.profit_factor(clean_returns)
            metrics['Common_Sense_Ratio'] = qs.stats.common_sense_ratio(clean_returns)
            metrics['Recovery_Factor'] = qs.stats.recovery_factor(clean_returns)
            metrics['CAGR'] = qs.stats.cagr(clean_returns)
            metrics['VaR'] = qs.stats.var(clean_returns)
            metrics['CVaR'] = qs.stats.cvar(clean_returns)
            metrics['Expected_Return'] = qs.stats.expected_return(clean_returns)
            metrics['Gain_To_Pain_Ratio'] = qs.stats.gain_to_pain_ratio(clean_returns)
            metrics['Tail_Ratio'] = qs.stats.tail_ratio(clean_returns)
            metrics['Outlier_Win_Ratio'] = qs.stats.outlier_win_ratio(clean_returns)
            metrics['Outlier_Loss_Ratio'] = qs.stats.outlier_loss_ratio(clean_returns)
            metrics['Win_Rate'] = qs.stats.win_rate(clean_returns)
            metrics['Avg_Win'] = qs.stats.avg_win(clean_returns)
            metrics['Avg_Loss'] = qs.stats.avg_loss(clean_returns)
            metrics['Best_Month'] = qs.stats.best(clean_returns)
            metrics['Worst_Month'] = qs.stats.worst(clean_returns)
            metrics['Consecutive_Wins'] = qs.stats.consecutive_wins(clean_returns)
            metrics['Consecutive_Losses'] = qs.stats.consecutive_losses(clean_returns)
            metrics['Exposure'] = qs.stats.exposure(clean_returns)
            metrics['Kelly_Criterion'] = qs.stats.kelly_criterion(clean_returns)
            metrics['Risk_Return_Ratio'] = qs.stats.risk_return_ratio(clean_returns)
            metrics['Risk_Of_Ruin'] = qs.stats.ror(clean_returns)

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {metric: np.nan for metric in self._get_metric_names()}
        
        return metrics
    
    def _get_metric_names(self) -> list:
        """Get list of all metric names."""
        return [
            'Cumulative_Return', 'Sharpe', 'Smart_Sharpe', 'Sortino', 'Smart_Sortino',
            'Max_Drawdown', 'Volatility_Ann', 'Calmar', 'Skew', 'Kurtosis',
            'Payoff_Ratio', 'Profit_Factor', 'Common_Sense_Ratio', 'Recovery_Factor',
            'CAGR', 'VaR', 'CVaR', 'Expected_Return', 'Gain_To_Pain_Ratio',
            'Tail_Ratio', 'Outlier_Win_Ratio', 'Outlier_Loss_Ratio',
            'Win_Rate', 'Avg_Win', 'Avg_Loss', 'Best_Month', 'Worst_Month',
            'Consecutive_Wins', 'Consecutive_Losses', 'Exposure', 'Kelly_Criterion',
            'Risk_Return_Ratio', 'Risk_Of_Ruin'
        ]
    
    def generate_rolling_features(self, monthly_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rolling economic indicator features.
        Creates two feature groups: Buy (using returns as-is) and Sell (using inverted returns).
        
        Parameters:
        -----------
        monthly_returns : pd.DataFrame
            DataFrame with monthly returns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with rolling economic indicator features for both buy and sell scenarios
        """
        feature_df = pd.DataFrame(index=monthly_returns.index)
        
        print(f"Generating rolling features with {self.time_window}-month window...")
        print(f"Starting with {len(monthly_returns)} monthly returns")
        
        processed_windows = 0
        skipped_windows = 0
        
        for i in range(self.time_window, len(monthly_returns)):
            current_date = monthly_returns.index[i]
            
            # Get returns for the current time window
            window_returns = monthly_returns.iloc[i-self.time_window:i]['Monthly_Return']
            
            # Skip if window has too few data points
            if len(window_returns.dropna()) < 2:
                skipped_windows += 1
                continue
            
            # Calculate metrics for Buy scenario (returns as-is, multiply by 1)
            buy_returns = window_returns * 1
            buy_metrics = self.calculate_performance_metrics(buy_returns)
            for metric_name, metric_value in buy_metrics.items():
                col_name = f'Buy_{metric_name}'
                if col_name not in feature_df.columns:
                    feature_df[col_name] = np.nan
                feature_df.loc[current_date, col_name] = metric_value
            
            # Calculate metrics for Sell scenario (inverted returns, multiply by -1)
            sell_returns = window_returns * -1
            sell_metrics = self.calculate_performance_metrics(sell_returns)
            for metric_name, metric_value in sell_metrics.items():
                col_name = f'Sell_{metric_name}'
                if col_name not in feature_df.columns:
                    feature_df[col_name] = np.nan
                feature_df.loc[current_date, col_name] = metric_value
            
            processed_windows += 1
        
        print(f"   Processed {processed_windows} windows, skipped {skipped_windows} windows with insufficient data")
        
        return feature_df

    def run_pipeline(self, save_results: bool = True) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline with binary predictions.
        
        Parameters:
        -----------
        save_results : bool
            Whether to save results to file
            
        Returns:
        --------
        pd.DataFrame
            Final feature set
        """
        print("=== Starting Economic Indicators Feature Engineering Pipeline (Binary Prediction) ===")
        
        # Load price data
        eurusd_data = self.load_eurusd_data()
        print(f"Loaded {len(eurusd_data)} daily price observations")
        
        # Generate monthly returns  
        monthly_returns = self.generate_monthly_returns(eurusd_data)
        print(f"Generated monthly returns for {len(monthly_returns)} observations")
        
        # Generate rolling features based on monthly returns
        features = self.generate_rolling_features(monthly_returns)
        print(f"Generated rolling features: {len(features)} rows x {len(features.columns)} columns")
        
        # Final cleanup
        #final_features = features.dropna()
        final_features = features
        print(f"Final feature set after removing NaN: {len(final_features)} rows x {len(final_features.columns)} columns")
        
        if save_results and len(final_features) > 0:
            output_file = 'economic_indicators_features.csv'
            final_features.to_csv(output_file)
            print(f"Features saved to {output_file}")
        
        print("\n=== Feature Engineering Pipeline Complete ===")
        
        return final_features
    
    def run_feature_engineering(self, eurusd_file_path: str, save_features: bool = True, 
                               output_file: str = "economic_indicators_features.csv") -> pd.DataFrame:
        """
        Compatibility wrapper for the main pipeline method.
        """
        # Update the file path
        self.eurusd_file_path = eurusd_file_path
        
        # Run pipeline and return features
        return self.run_pipeline(save_results=save_features)
def main():
    """Main function to run the feature engineering pipeline."""
    
    # Initialize feature engineering class
    feature_engineer = EconomicIndicatorsFeatureEngineering(
        time_window=3              # 3-month rolling window
    )
    
    # Run the complete pipeline
    try:
        features = feature_engineer.run_feature_engineering(
            eurusd_file_path="EURUSD.csv",
            save_features=True,
            output_file="economic_indicators_features.csv"
        )
        
        print("\n=== Summary Statistics ===")
        print(features.describe())
        
        print(f"\n=== Sample Features (Last 5 rows) ===")
        print(features.tail())
        
    except Exception as e:
        print(f"Error in feature engineering pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
