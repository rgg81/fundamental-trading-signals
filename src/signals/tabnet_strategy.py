import pandas as pd
import numpy as np
import optuna
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import torch
from strategy import Strategy
import warnings
warnings.filterwarnings('ignore')

class TabNetOptunaStrategy(Strategy):
    def __init__(self, n_trials=3, n_splits=5, random_state=42, use_gpu=True):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.fitted = False
        self.use_gpu = use_gpu
        
        # Set device for GPU/CPU usage
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("Using CPU")
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _clean_data(self, df):
        """Clean data by removing rows with NaN or infinity values"""
        # Make a copy to avoid modifying the original data
        df_clean = df.copy()
        
        # Remove rows with infinity values
        df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
        
        # Remove rows with NaN values
        df_clean = df_clean.dropna()
        
        return df_clean

    def _prepare_data(self, X, y=None):
        """Prepare data for TabNet (convert to numpy arrays with proper dtypes)"""
        # Convert to numpy arrays
        X_array = X.values.astype(np.float32)
        
        if y is not None:
            y_array = y.values.astype(np.int64)
            return X_array, y_array
        
        return X_array

    def fit(self, X, y):
        # Clean data to handle infinities
        X_clean = self._clean_data(X)
        
        def objective(trial):
            # Define the hyperparameter search space for TabNet
            params = {
              #  'n_d': trial.suggest_int('n_d', 8, 64),  # Width of the decision prediction layer
              #  'n_a': trial.suggest_int('n_a', 8, 64),  # Width of the attention embedding for each mask
             #   'n_steps': trial.suggest_int('n_steps', 3, 10),  # Number of steps in the architecture
              #  'gamma': trial.suggest_float('gamma', 1.0, 2.0),  # Coefficient for feature reusage in the masks
              #  'n_independent': trial.suggest_int('n_independent', 1, 5),  # Number of independent Glu layers at each step
              #  'n_shared': trial.suggest_int('n_shared', 1, 5),  # Number of shared Glu layers at each step
              #  'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),  # Sparsity regularization
              #  'momentum': trial.suggest_float('momentum', 0.01, 0.4),  # Momentum for batch normalization
              #  'clip_value': trial.suggest_float('clip_value', 1.0, 2.0),  # Gradient clipping value
                'lr': trial.suggest_float('lr', 1e-3, 1e-1, log=True),  # Learning rate
                'max_epochs': trial.suggest_int('max_epochs', 100, 500),  # Maximum number of epochs
               # 'patience': trial.suggest_int('patience', 10, 50),  # Early stopping patience
                'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 2048]),  # Batch size
            }
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_clean):
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    # Prepare data for TabNet
                    X_train_array, y_train_array = self._prepare_data(X_train, y_train)
                    X_val_array, y_val_array = self._prepare_data(X_val, y_val)
                    
                    # Create TabNet classifier
                    tabnet = TabNetClassifier(
                     #   n_d=params['n_d'],
                     #   n_a=params['n_a'],
                     #   n_steps=params['n_steps'],
                     #   gamma=params['gamma'],
                     #   n_independent=params['n_independent'],
                     #   n_shared=params['n_shared'],
                     #   lambda_sparse=params['lambda_sparse'],
                     #   momentum=params['momentum'],
                     #   clip_value=params['clip_value'],
                        optimizer_params=dict(lr=params['lr']),
                        scheduler_params={"step_size": 50, "gamma": 0.9},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        mask_type='entmax',  # Type of attention mask
                        device_name=self.device,  # GPU/CPU device
                   #     seed=self.random_state,
                        verbose=0
                    )
                    
                    # Fit the model
                    tabnet.fit(
                        X_train_array, y_train_array,
                    #    eval_set=[(X_val_array, y_val_array)],
                        max_epochs=params['max_epochs'],
                   #     patience=params['patience'],
                        batch_size=params['batch_size'],
                        virtual_batch_size=params['batch_size'] // 4,  # Usually 1/4 of batch_size
                        num_workers=0,
                        drop_last=False,
                        eval_metric=['accuracy']
                    )
                    
                    # Make predictions
                    preds = tabnet.predict(X_val_array)
                    scores.append(accuracy_score(y_val_array, preds))
                    
                except Exception as e:
                    print(f"Error in TabNet training: {e}")
                    return 0.0  # Return worst score for failed trials
                    
            return 1.0 - np.mean(scores)  # minimize error

        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        print(f"Best TabNet parameters: {self.best_params}")
        
        # Train the final model with best parameters
        X_array, y_array = self._prepare_data(X_clean, y)
        
        self.model = TabNetClassifier(
        #    n_d=self.best_params['n_d'],
        #    n_a=self.best_params['n_a'],
         #   n_steps=self.best_params['n_steps'],
        #    gamma=self.best_params['gamma'],
        #    n_independent=self.best_params['n_independent'],
        #    n_shared=self.best_params['n_shared'],
        #    lambda_sparse=self.best_params['lambda_sparse'],
        #    momentum=self.best_params['momentum'],
        #    clip_value=self.best_params['clip_value'],
            optimizer_params=dict(lr=self.best_params['lr']),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            device_name=self.device,
          #  seed=self.random_state,
            verbose=0
        )
        
        # Fit the final model
        self.model.fit(
            X_array, y_array,
            max_epochs=self.best_params['max_epochs'],
          #  patience=self.best_params['patience'],
            batch_size=self.best_params['batch_size'],
            virtual_batch_size=self.best_params['batch_size'] // 4,
            num_workers=0,
            drop_last=False
        )
        
        self.fitted = True
        
        # Evaluate on training data
        preds = self.model.predict(X_array)
        print("Classification Report:\n", classification_report(y_array, preds))

        # Feature importance for TabNet (using attention masks)
        try:
            # Get feature importance from TabNet
            feature_importance_matrix = self.model.feature_importances_
            
            # Average importance across all steps
            feature_importance = np.mean(feature_importance_matrix, axis=0)
            
            feature_importance_df = pd.DataFrame({
                'Feature': X_clean.columns,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 most important features (TabNet attention):")
            print(feature_importance_df.head(10))
            
            # Global feature importance (accumulated across all steps)
            global_importance = np.sum(feature_importance_matrix, axis=0)
            global_importance_df = pd.DataFrame({
                'Feature': X_clean.columns,
                'Global_Importance': global_importance
            }).sort_values('Global_Importance', ascending=False)
            
            print("\nTop 10 features by global importance:")
            print(global_importance_df.head(10))
            
        except Exception as e:
            print(f"Could not get TabNet feature importance: {e}")

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
        
        # Prepare prediction data
        X_pred_array = self._prepare_data(X_pred_clean)
        
        # Predict
        pred = self.model.predict(X_pred_array)[0]
        
        # Get prediction probabilities
        pred_probs = self.model.predict_proba(X_pred_array)[0]
        print(f"Prediction probabilities: Class 0: {pred_probs[0]:.4f}, Class 1: {pred_probs[1]:.4f}")
        
        # TabNet provides additional interpretability - get local feature importance
        try:
            # Get explain matrix for the prediction (local interpretability)
            explain_matrix, masks = self.model.explain(X_pred_array)
            print(f"TabNet decision process used {len(masks)} attention steps")
            
            # Show most important features for this specific prediction
            if len(explain_matrix) > 0:
                feature_contribution = explain_matrix[0]  # For the first (and likely only) sample
                top_features_idx = np.argsort(np.abs(feature_contribution))[-3:]  # Top 3 features
                
                print("Top 3 features influencing this prediction:")
                for idx in reversed(top_features_idx):
                    feature_name = X_pred_clean.columns[idx]
                    contribution = feature_contribution[idx]
                    print(f"  {feature_name}: {contribution:.4f}")
                    
        except Exception as e:
            print(f"Could not get local feature importance: {e}")
        
        return int(pred), 10  # signal, amount
