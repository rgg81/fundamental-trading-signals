import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
from strategy import Strategy

warnings.filterwarnings('ignore')

class VotingEnsembleStrategy(Strategy):
    """
    A strategy that builds 10 different models and uses a voting system 
    to determine the final trading signal.
    """
    
    def __init__(self, voting_method='majority', n_splits=5, threshold=0.5, random_state=42, close_col='EURUSD_Close'):
        """
        Initialize the voting ensemble strategy.
        
        Parameters:
        -----------
        voting_method : str
            'majority' for majority voting, 'weighted' for weighted voting based on validation score
        n_splits : int
            Number of splits for time series cross-validation
        threshold : float
            Threshold for probability-based decisions (between 0 and 1)
        random_state : int
            Random seed for reproducibility
        """
        self.voting_method = voting_method
        self.n_splits = n_splits
        self.threshold = threshold
        self.random_state = random_state
        self.models = []
        self.weights = []
        self.fitted = False
        self.scaler = RobustScaler()  # Changed to RobustScaler to handle outliers better
        self.imputer = SimpleImputer(strategy='median')  # Added imputer to handle missing values
        
    def _preprocess_data(self, X):
        """
        Preprocess data to handle missing values, infinity, and outliers.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        X_processed : pd.DataFrame
            Processed feature matrix
        """
        # Make a copy to avoid modifying the original data
        X_processed = X.copy()
        
        # Handle infinity
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many missing values (> 50%)
        missing_percent = X_processed.isna().mean()
        cols_to_drop = missing_percent[missing_percent > 0.5].index
        
        if len(cols_to_drop) > 0:
            print(f"Dropping columns with >50% missing values: {list(cols_to_drop)}")
            X_processed = X_processed.drop(columns=cols_to_drop)
            
        if X_processed.empty:
            raise ValueError("All features have been dropped due to missing values. Check your data.")
        
        # Impute remaining missing values
        X_processed = pd.DataFrame(
            self.imputer.fit_transform(X_processed),
            index=X_processed.index,
            columns=X_processed.columns
        )
        
        # Apply scaling
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_processed),
            index=X_processed.index,
            columns=X_processed.columns
        )
        
        return X_scaled
    
    def _create_models(self):
        """Create 10 different models with various algorithms"""
        models = [
           # ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
           # ('lgbm', lgb.LGBMClassifier(random_state=self.random_state, n_estimators=5, max_depth=5, colsample_bytree=0.5, subsample=0.5)),
          #  ('gb', GradientBoostingClassifier(random_state=self.random_state)),
           # ('svm', SVC(probability=True, random_state=self.random_state)),
            ('mlp', MLPClassifier(max_iter=1000, random_state=self.random_state)),
            ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000)),
            ('nb', GaussianNB()),
          #  ('knn', KNeighborsClassifier()),
          #  ('dt', DecisionTreeClassifier(random_state=self.random_state)),
          #  ('ab', AdaBoostClassifier(random_state=self.random_state))
        ]
        return models
    
    def fit(self, X, y):
        """
        Train all models and compute their weights.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target labels (0 for sell, 1 for buy)
        """
        # Preprocess data
        try:
            X_scaled = self._preprocess_data(X)
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            # Try a simpler approach if preprocessing fails
            print("Attempting simpler preprocessing...")
            # Replace inf values and drop columns with all NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna(axis=1, how='all')
            
            # If there are still NaN values, drop rows or fill with median
            if X.isna().any().any():
                X = X.fillna(X.median())
                
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
            
        # Create models
        model_list = self._create_models()
        self.models = []
        self.weights = []
        
        # Time series cross-validation for weight calculation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for name, model in model_list:
            try:
                # Train the model
                print(f"Training {name} model...")
                model.fit(X_scaled, y)
                self.models.append((name, model))
                
                # Calculate validation score for weights
                scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model_cv = type(model)(**model.get_params())
                    model_cv.fit(X_train, y_train)
                    
                    try:
                        # Prefer ROC AUC if the model supports predict_proba
                        if hasattr(model_cv, 'predict_proba'):
                            y_prob = model_cv.predict_proba(X_val)[:, 1]
                            score = roc_auc_score(y_val, y_prob)
                        else:
                            y_pred = model_cv.predict(X_val)
                            score = accuracy_score(y_val, y_pred)
                        scores.append(score)
                    except Exception:
                        # Fallback to accuracy if ROC AUC fails
                        y_pred = model_cv.predict(X_val)
                        score = accuracy_score(y_val, y_pred)
                        scores.append(score)
                
                # Store weight as average score across folds
                weight = np.mean(scores)
                self.weights.append(weight)
                print(f"{name} model weight: {weight:.4f}")
            except Exception as e:
                print(f"Error training {name} model: {e}")
                # Skip this model
                continue
                
        # Check if any models were successfully trained
        if not self.models:
            raise ValueError("No models could be trained successfully. Check your data.")
            
        # Normalize weights
        self.weights = np.array(self.weights)
        if len(self.weights) > 0:
            self.weights = self.weights / np.sum(self.weights)
        
        self.fitted = True
        print("Ensemble model training complete.")
        
    def predict(self, X):
        """
        Generate predictions for a feature matrix.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        
        Returns:
        --------
        predictions : np.array
            Array of binary predictions (0 for sell, 1 for buy)
        """
        if not self.fitted:
            raise Exception("Models not trained. Call fit(X, y) first.")
        
        # Clean and scale input data
        try:
            # Handle columns that might be in X but not in the training data
            X = X[self.scaler.feature_names_in_]
        except (AttributeError, KeyError):
            # If feature_names_in_ is not available or columns don't match
            # Just continue and let preprocessing handle it
            pass
        
        # Preprocess data
        try:
            # Replace infinities with NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN with median values from training
            X = self.imputer.transform(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            print(f"Error preprocessing prediction data: {e}")
            # Return a default prediction if preprocessing fails
            return np.zeros(len(X))
            
        if self.voting_method == 'majority':
            # Get predictions from each model
            predictions = []
            for _, model in self.models:
                try:
                    if hasattr(model, 'predict_proba'):
                        pred_prob = model.predict_proba(X_scaled)[:, 1]
                        pred = (pred_prob > self.threshold).astype(int)
                    else:
                        pred = model.predict(X_scaled)
                    predictions.append(pred)
                except Exception as e:
                    print(f"Error getting predictions from model: {e}")
                    # Skip this model if it fails
                    continue
            
            if not predictions:
                # If all models failed, return default prediction
                return np.zeros(X_scaled.shape[0])
                
            # Stack predictions and find the majority vote
            predictions = np.vstack(predictions)
            final_pred = np.mean(predictions, axis=0) > 0.5
            return final_pred.astype(int)
        
        elif self.voting_method == 'weighted':
            # Get predictions from each model and apply weights
            weighted_preds = np.zeros(X_scaled.shape[0])
            
            for i, (_, model) in enumerate(self.models):
                try:
                    if hasattr(model, 'predict_proba'):
                        pred_prob = model.predict_proba(X_scaled)[:, 1]
                        weighted_preds += pred_prob * self.weights[i]
                    else:
                        pred = model.predict(X_scaled)
                        weighted_preds += pred * self.weights[i]
                except Exception as e:
                    print(f"Error getting weighted predictions: {e}")
                    # Skip this model if it fails
                    continue
            
            # Apply threshold to get final prediction
            final_pred = (weighted_preds > self.threshold).astype(int)
            return final_pred
        
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
    
    def generate_signal(self, past_data, current_data):
        """
        Generate a trading signal based on model predictions.
        
        Parameters:
        -----------
        past_data : pd.DataFrame
            Historical data for model training
        current_data : pd.DataFrame or pd.Series
            Current market data for prediction
        
        Returns:
        --------
        signal : int
            1 for buy signal, 0 for sell/hold
        amount : int
            Trading amount (fixed at 10)
        """
        # X all data frame less Label, Date and Close
        columns_to_drop = ['Label', 'Date', 'EURUSD_Close']
        X = past_data.drop(columns=columns_to_drop)
        # y only Label
        y = past_data['Label']
        # Fit model
        # I want to reset the model every time I call generate_signal
        self.model = None
        self.fit(X, y)
        # X_pred should be the same features as X
        X_pred = current_data.drop(columns=columns_to_drop)
        # current_data should be a pandas Series or DataFrame with the same features as X
        pred = self.predict(X_pred)[0]
        return int(pred), 10  # signal, amount