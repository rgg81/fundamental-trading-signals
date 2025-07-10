import pandas as pd
import numpy as np
import optuna
from ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Bernoulli
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from strategy import Strategy

class NGBoostOptunaStrategy(Strategy):
    def __init__(self, n_trials=10, n_splits=5, random_state=42):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.fitted = False

    def _clean_data(self, df):
        """Clean data by removing rows with NaN or infinity values"""
        # Make a copy to avoid modifying the original data
        df_clean = df.copy()
        
        # Remove rows with infinity values
        df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
        
        # Remove rows with NaN values
        df_clean = df_clean.dropna()
        
        return df_clean
    
    def _validate_data(self, X, y):
        """Validate data for NGBoost training"""
        # Check if we have enough samples
        if len(X) < 20:
            raise ValueError(f"Insufficient data: {len(X)} samples, need at least 20")
        
        # Check if we have both classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(f"Need both classes for classification, found: {unique_classes}")
        
        # Check class balance - if too imbalanced, it might cause issues
        class_counts = np.bincount(y)
        minority_ratio = min(class_counts) / len(y)
        if minority_ratio < 0.05:  # Less than 5% minority class
            print(f"Warning: Imbalanced classes detected. Minority class ratio: {minority_ratio:.3f}")
        
        # Check for constant features
        feature_vars = np.var(X, axis=0)
        zero_var_features = np.sum(feature_vars == 0)
        if zero_var_features > 0:
            print(f"Warning: {zero_var_features} constant features detected")
        
        return True

    def fit(self, X, y):
        # Clean data to handle infinities
        X_clean = self._clean_data(X)
        
        # Validate data before proceeding
        try:
            self._validate_data(X_clean.values, y.values)
        except ValueError as e:
            print(f"Data validation failed: {e}")
            # Create a simple fallback model
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression()
            self.model.fit(X_clean, y)
            self.fitted = True
            self.best_params = {"fallback": "LogisticRegression"}
            print("Using LogisticRegression fallback due to data issues")
            return
        
        def objective(trial):
            # Define the hyperparameter search space for NGBoost
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),  # Reduced range
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),  # Reduced range
               # 'minibatch_frac': trial.suggest_float('minibatch_frac', 0.8, 1.0),  # Higher minibatch
                'col_sample': trial.suggest_float('col_sample', 0.5, 1.0),  # Higher col_sample
                #'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True),  # More lenient tolerance
                #'random_state': self.random_state,
                'verbose': False
            }
            
            # Tree-specific parameters - more conservative
            tree_params = {
                'criterion': trial.suggest_categorical('criterion', ['friedman_mse']),  # More stable
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),  # Higher minimum
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 10),  # Higher minimum
                'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.1),  # Lower max
                'max_depth': trial.suggest_int('max_depth', 1, 2),  # Shallower trees
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])  # Remove None
            }
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            for train_idx, val_idx in tscv.split(X_clean):
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    # Additional validation for each fold
                    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                        continue  # Skip this fold if missing classes
                    
                    # Create base learner with tree parameters
                    base_learner = DecisionTreeRegressor(**tree_params)
                    
                    # Create NGBoost classifier
                    ngb = NGBoost(
                        Base=base_learner,
                        Dist=Bernoulli,
                        Score=LogScore,
                        **params
                    )
                    
                    # Convert to numpy arrays (NGBoost prefers numpy)
                    X_train_np = X_train.values
                    X_val_np = X_val.values
                    y_train_np = y_train.values
                    y_val_np = y_val.values
                    
                    # Fit the model with error handling
                    ngb.fit(X_train_np, y_train_np, 
                           X_val=X_val_np, Y_val=y_val_np,
                           early_stopping_rounds=30)  # Reduced from 50
                    
                    # Make predictions
                    preds = ngb.predict(X_val_np)
                    scores.append(accuracy_score(y_val_np, preds))
                    
                except Exception as e:
                    print(f"Error in NGBoost training fold: {e}")
                    continue  # Skip this fold instead of failing entirely
                    
            if not scores:  # No successful folds
                return 1.0  # Return worst score
                
            return 1.0 - np.mean(scores)  # minimize error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        print(f"Best NGBoost parameters: {self.best_params}")
        
        # Extract tree parameters from best_params
        tree_params = {
            'criterion': self.best_params.get('criterion', 'friedman_mse'),
            'min_samples_split': self.best_params.get('min_samples_split', 5),
            'min_samples_leaf': self.best_params.get('min_samples_leaf', 3),
            'min_weight_fraction_leaf': self.best_params.get('min_weight_fraction_leaf', 0.0),
            'max_depth': self.best_params.get('max_depth', 3),
            'max_features': self.best_params.get('max_features', 'sqrt')
        }
        
        # Extract NGBoost parameters
        ngb_params = {k: v for k, v in self.best_params.items() if k not in tree_params}
        
        # Train the final model with best parameters and error handling
        try:
            base_learner = DecisionTreeRegressor(**tree_params)
            self.model = NGBoost(
                Base=base_learner,
                Dist=Bernoulli,
                Score=LogScore,
                **ngb_params
            )
            
            # Convert to numpy arrays
            X_clean_np = X_clean.values
            y_np = y.values
            
            self.model.fit(X_clean_np, y_np)
            self.fitted = True
            
            # Evaluate on training data
            preds = self.model.predict(X_clean_np)
            print("Classification Report:\n", classification_report(y_np, preds))
            
        except Exception as e:
            print(f"Error in final NGBoost training: {e}")
            print("Falling back to LogisticRegression")
            # Fallback to a simple model
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(random_state=self.random_state)
            self.model.fit(X_clean, y)
            self.fitted = True
            self.best_params = {"fallback": "LogisticRegression"}


    def generate_signal(self, past_data, current_data):
        # Clean input data
        past_data = self._clean_data(past_data)
        current_data = self._clean_data(current_data)

        if current_data.empty:
            return None, 10
        
        columns_to_drop = ['Label', 'Date', 'EURUSD_Close']
        X = past_data.drop(columns=columns_to_drop)
        # y only Label
        y = past_data['Label']
        
        # Fit model
        # Reset the model every time we call generate_signal
        self.model = None
        self.fit(X, y)
        
        # X_pred should be the same features as X
        X_pred = current_data.drop(columns=columns_to_drop)
        
        # Clean prediction data
        X_pred_clean = self._clean_data(X_pred)
        
        # Convert to numpy array
        X_pred_np = X_pred_clean.values
        
        # Predict with error handling
        try:
            pred = self.model.predict(X_pred_np)[0]
            
            # Handle different model types
            if hasattr(self.model, 'predict_proba'):
                # LogisticRegression fallback
                pred_probs = self.model.predict_proba(X_pred_np)[0]
                prob_class_0 = pred_probs[0]
                prob_class_1 = pred_probs[1]
                pred_uncertainty = min(prob_class_0, prob_class_1)  # Uncertainty is min probability
                
                print(f"Prediction probabilities: Class 0: {prob_class_0:.4f}, Class 1: {prob_class_1:.4f}")
                print(f"Prediction uncertainty: {pred_uncertainty:.4f}")
                
                if pred_uncertainty > 0.4:  # High uncertainty for logistic regression
                    print("High uncertainty detected - consider reducing position size")
                    
            else:
                # NGBoost model
                print(f"NGBoost prediction: {pred}")
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Return random prediction as fallback
            pred = np.random.randint(0, 2)
            print(f"Using random fallback prediction: {pred}")
        
        return int(pred), 10  # signal, amount
