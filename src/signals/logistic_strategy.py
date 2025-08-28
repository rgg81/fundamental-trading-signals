import pandas as pd
import numpy as np
import optuna
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from strategy import Strategy

class LogisticRegressionOptunaStrategy(Strategy):
    def __init__(self, n_trials=5, n_splits=5, random_state=42):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.fitted = False
        self.scaler = StandardScaler()

    def _clean_data(self, X):
        """Clean data by handling NaN and infinity values"""
        # Make a copy to avoid modifying the original data
        X_clean = X.copy()
        
        # Print initial stats about problematic values
        inf_count = np.isinf(X_clean.values).sum()
        nan_count = np.isnan(X_clean.values).sum()
        if inf_count > 0 or nan_count > 0:
            print(f"Found {inf_count} infinite values and {nan_count} NaN values in the data")
        
        # Replace infinity with NaN first
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values by filling with column median
        X_clean = X_clean.fillna(X_clean.median())
        
        # If any NaN values remain (could happen if entire column is NaN), fill with 0
        if X_clean.isna().any().any():
            X_clean = X_clean.fillna(0)
            
        # Clip extremely large values to prevent numerical issues
        # Get the 0.1% and 99.9% percentiles for each column
        lower_bounds = X_clean.quantile(0.001)
        upper_bounds = X_clean.quantile(0.999)
        
        # Clip values outside this range
        for col in X_clean.columns:
            X_clean[col] = np.clip(X_clean[col], lower_bounds[col], upper_bounds[col])
            
        return X_clean

    def fit(self, X, y):
        # Clean data before training
        X_clean = self._clean_data(X)
        X_scaled = self.scaler.fit_transform(X_clean)
        X_scaled_df = pd.DataFrame(X_scaled, index=X_clean.index, columns=X_clean.columns)
        
        def objective(trial):
            # Define regularization approach
            reg_type = trial.suggest_categorical('reg_type', ['elasticnet'])
            
            # Base parameters
            params = {
                'max_iter': trial.suggest_int('max_iter', 400, 1000),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                #'random_state': self.random_state,
                'solver': 'saga',  # saga supports all penalties
            }
            
            # Set regularization parameters based on type
            if reg_type == 'none':
                params['penalty'] = None  # Using None instead of 'none'
            else:
                params['penalty'] = reg_type
                
                # Inverse of regularization strength (higher C = less regularization)
                params['C'] = trial.suggest_float('C', 0.01, 1, log=True)
                
                # For elasticnet, tune the mix of L1 and L2
                if reg_type == 'elasticnet':
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                    print(f"Testing elasticnet with l1_ratio={params['l1_ratio']} (0=L2 only, 1=L1 only)")
                elif reg_type == 'l1':
                    print(f"Testing L1 regularization with C={params['C']}")
                elif reg_type == 'l2':
                    print(f"Testing L2 regularization with C={params['C']}")
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled_df):
                X_train, X_val = X_scaled_df.iloc[train_idx], X_scaled_df.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    # Create and train the model
                    lr = LogisticRegression(**params)
                    lr.fit(X_train, y_train)
                    preds = lr.predict(X_val)
                    scores.append(accuracy_score(y_val, preds))
                except Exception as e:
                    print(f"Error in Logistic Regression training: {e}")
                    return 0.0  # Return worst score for failed trials
                    
            return 1.0 - np.mean(scores)  # minimize error

        # Create and run the optimization study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get the best parameters
        self.best_params = study.best_params
        
        # Process the parameters for the final model
        final_params = {
            'max_iter': self.best_params['max_iter'],
            'class_weight': self.best_params['class_weight'],
            #'random_state': self.random_state,
            'solver': 'saga'
        }
        
        # Setup regularization based on best params
        reg_type = self.best_params['reg_type']
        if reg_type == 'none':
            final_params['penalty'] = None  # Using None instead of 'none'
        else:
            final_params['penalty'] = reg_type
            final_params['C'] = self.best_params['C']
            
            if reg_type == 'elasticnet':
                final_params['l1_ratio'] = self.best_params['l1_ratio']
                
        print(f"Best Logistic Regression parameters: {final_params}")
        
        # Train the final model with best parameters
        self.model = LogisticRegression(**final_params)
        self.model.fit(X_scaled_df, y)
        self.fitted = True
        
        # Print classification report and feature importance
        preds = self.model.predict(X_scaled_df)
        print("Classification Report:\n", classification_report(y, preds))
        
        # Print feature importance (coefficients) if applicable
        if reg_type != 'none':
            # Get feature importance
            feature_importance = pd.DataFrame({
                'Feature': X_clean.columns,
                'Importance': np.abs(self.model.coef_[0])
            }).sort_values('Importance', ascending=False)
            print("\nTop 10 most important features:")
            print(feature_importance.head(10))

    def generate_signal(self, past_data, current_data):
        # X all data frame less Label, Date and Close
        columns_to_drop = ['Label', 'Date', 'EURUSD_Close']
        
        X = past_data.drop(columns=columns_to_drop)
        # y only Label
        y = past_data['Label']
        
        # Fit model
        # Reset the model every time we call generate_signal
        self.model = None
        self.scaler = StandardScaler()  
        self.fit(X, y)

        X_pred = current_data.drop(columns=columns_to_drop)
        
        # Clean prediction data
        X_pred_clean = self._clean_data(X_pred)
         # Normalize the clean prediction data using the same scaler
        X_pred_scaled = self.scaler.transform(X_pred_clean)
        
        # Convert back to DataFrame to preserve feature names
        X_pred_scaled_df = pd.DataFrame(X_pred_scaled, index=X_pred_clean.index, columns=X_pred_clean.columns)
        
        # Predict signal
        pred = self.model.predict(X_pred_scaled_df)[0]
        return int(pred), 10  # signal, amount