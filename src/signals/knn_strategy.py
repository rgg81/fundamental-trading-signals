import pandas as pd
import numpy as np
import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from strategy import Strategy

class KNNOptunaStrategy(Strategy):
    def __init__(self, n_trials=10, n_splits=5, random_state=42):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.fitted = False
        self.scaler = StandardScaler()  # Add scaler for normalization

    def _clean_data(self, X):
        """Clean data by handling NaN and infinity values"""
        # Make a copy to avoid modifying the original data
        X_clean = X.copy()
        
        # Handle infinity values - replace with large but finite values
        X_clean = X_clean.replace([np.inf], 1e30)
        X_clean = X_clean.replace([-np.inf], -1e30)
        
        # Check if there are still any infinity values
        if np.isinf(X_clean.values).any():
            inf_mask = np.isinf(X_clean.values)
            inf_count = inf_mask.sum()
            print(f"Replacing {inf_count} remaining infinity values")
            X_clean.values[inf_mask] = np.sign(X_clean.values[inf_mask]) * 1e30
        
        # Handle NaN values by filling with column median
        X_clean = X_clean.fillna(X_clean.median())
        
        # If any NaN values remain (could happen if entire column is NaN), fill with 0
        if X_clean.isna().any().any():
            X_clean = X_clean.fillna(0)
            
        return X_clean

    def fit(self, X, y):
        # Clean data to handle infinities
        X_clean = self._clean_data(X)
        
        # Normalize the cleaned input data
        #X_scaled = self.scaler.fit_transform(X_clean)
        #X_scaled_df = pd.DataFrame(X_scaled, index=X_clean.index, columns=X_clean.columns)
        X_scaled_df = X_clean
        
        def objective(trial):
            # Define the hyperparameter search space for KNN
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
                #'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                #'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                #'leaf_size': trial.suggest_int('leaf_size', 10, 50),
                # 'p': trial.suggest_int('p', 1, 2),  # p=1 is manhattan_distance, p=2 is euclidean_distance
            }
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled_df):
                X_train, X_val = X_scaled_df.iloc[train_idx], X_scaled_df.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    knn = KNeighborsClassifier(**params)
                    knn.fit(X_train, y_train)
                    preds = knn.predict(X_val)
                    scores.append(accuracy_score(y_val, preds))
                except Exception as e:
                    print(f"Error in KNN training: {e}")
                    return 0.0  # Return worst score for failed trials
                    
            return 1.0 - np.mean(scores)  # minimize error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        print(f"Best KNN parameters: {self.best_params}")
        
        # Train the final model with best parameters
        self.model = KNeighborsClassifier(**self.best_params)
        self.model.fit(X_scaled_df, y)
        self.fitted = True
        
        # Evaluate on training data
        preds = self.model.predict(X_scaled_df)
        print("Classification Report:\n", classification_report(y, preds))
        
        # Analyze feature importance by evaluating each feature individually
        if X_clean.shape[1] > 1:  # Only if we have more than one feature
            importances = []
            for i in range(X_clean.shape[1]):
                # Create a KNN with just this feature
                single_feat_knn = KNeighborsClassifier(**self.best_params)
                X_feat = X_scaled_df.iloc[:, [i]]
                single_feat_knn.fit(X_feat, y)
                preds = single_feat_knn.predict(X_feat)
                score = accuracy_score(y, preds)
                importances.append(score)
            
            # Create a feature importance DataFrame
            feature_importance = pd.DataFrame({
                'Feature': X_clean.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 most important features (based on individual predictive power):")
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
        self.scaler = StandardScaler()  # Reset the scaler
        self.fit(X, y)
        
        # X_pred should be the same features as X
        X_pred = current_data.drop(columns=columns_to_drop)
        
        # Clean prediction data
        X_pred_clean = self._clean_data(X_pred)
        
        # Normalize the clean prediction data using the same scaler
        #X_pred_scaled = self.scaler.transform(X_pred_clean)
        
        # Convert back to DataFrame to preserve feature names
        #X_pred_scaled_df = pd.DataFrame(X_pred_scaled, index=X_pred_clean.index, columns=X_pred_clean.columns)

        X_pred_scaled_df = X_pred_clean
        
        # Get prediction and neighbors
        pred = self.model.predict(X_pred_scaled_df)[0]
        
        # Get prediction probabilities (if available)
        if hasattr(self.model, 'predict_proba'):
            pred_probs = self.model.predict_proba(X_pred_scaled_df)[0]
            print(f"Prediction probabilities: Class 0: {pred_probs[0]:.4f}, Class 1: {pred_probs[1]:.4f}")
        
        # Get the nearest neighbors for additional insight
        distances, indices = self.model.kneighbors(X_pred_scaled_df, n_neighbors=min(5, len(X)))
        print(f"Nearest neighbor distances: {distances[0]}")
        print(f"Nearest neighbor indices: {indices[0]}")
        
        return int(pred), 10  # signal, amount