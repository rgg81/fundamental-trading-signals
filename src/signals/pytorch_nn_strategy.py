import pandas as pd
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from strategy import Strategy

class MLPNet(nn.Module):
    """Simple multi-layer perceptron network optimized for small datasets"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.5, activation='relu', use_batch_norm=False):
        super(MLPNet, self).__init__()
        
        # Build layers dynamically - simpler architecture for small data
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))        
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            # Higher dropout for small datasets to prevent overfitting
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'selu':
            return nn.SELU()
        else:
            return nn.ReLU()  # Default
    
    def forward(self, x):
        return self.network(x)

class PyTorchNeuralNetOptunaStrategy(Strategy):
    def __init__(self, n_trials=3, n_splits=8, random_state=42, balance_classes=True, balance_method='undersample'):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.balance_classes = balance_classes
        self.balance_method = balance_method
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.best_params = None
        self.fitted = False
        
        # Set random seeds for reproducibility
        #torch.manual_seed(random_state)
        #np.random.seed(random_state)

    def _clean_data(self, df):
        """Clean data by removing rows with NaN or infinity values"""
        # Make a copy to avoid modifying the original data
        df_clean = df.copy()
        
        # Remove rows with infinity values
        df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
        
        # Remove rows with NaN values
        df_clean = df_clean.dropna()
        
        return df_clean

    def _balance_classes(self, X, y, method='undersample', random_state=None):
        """Balance classes in the dataset
        
        Args:
            X: Feature dataframe
            y: Target series
            method: 'undersample' or 'oversample'
            random_state: Random state for reproducibility
            
        Returns:
            X_balanced, y_balanced: Balanced feature and target data
        """
        if random_state is None:
            random_state = self.random_state
            
        # Get class counts
        class_counts = y.value_counts()
        print(f"Original class distribution: {class_counts.to_dict()}")
        
        # If classes are already balanced (within 10% difference), return original data
        min_class_count = min(class_counts)
        max_class_count = max(class_counts)
        
        if method == 'undersample':
            # Undersample majority class to match minority class
            target_count = min_class_count
            print(f"Undersampling to {target_count} samples per class")
            
            # Get indices for each class
            class_indices = {}
            for class_label in class_counts.index:
                class_indices[class_label] = y[y == class_label].index.tolist()
            
            # Sample from each class
            balanced_indices = []
            # np.random.seed(random_state)
            
            for class_label, indices in class_indices.items():
                if len(indices) > target_count:
                    # Randomly sample target_count indices
                    sampled_indices = np.random.choice(indices, target_count, replace=False)
                    balanced_indices.extend(sampled_indices)
                else:
                    # Use all available indices
                    balanced_indices.extend(indices)
            
            # Create balanced dataset
            X_balanced = X.loc[balanced_indices]
            y_balanced = y.loc[balanced_indices]
            
        elif method == 'oversample':
            # Oversample minority class to match majority class
            target_count = max_class_count
            print(f"Oversampling to {target_count} samples per class")
            
            # Get indices for each class
            class_indices = {}
            for class_label in class_counts.index:
                class_indices[class_label] = y[y == class_label].index.tolist()
            
            # Sample from each class
            balanced_indices = []
            # np.random.seed(random_state)
            
            for class_label, indices in class_indices.items():
                if len(indices) < target_count:
                    # Oversample with replacement
                    sampled_indices = np.random.choice(indices, target_count, replace=True)
                    balanced_indices.extend(sampled_indices)
                else:
                    # Use all available indices
                    balanced_indices.extend(indices)
            
            # Create balanced dataset
            X_balanced = X.loc[balanced_indices]
            y_balanced = y.loc[balanced_indices]
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Shuffle the balanced data
        combined_indices = list(range(len(X_balanced)))
       # np.random.seed(random_state)
        np.random.shuffle(combined_indices)
        
        X_balanced = X_balanced.iloc[combined_indices].reset_index(drop=True)
        y_balanced = y_balanced.iloc[combined_indices].reset_index(drop=True)
        
        # Print final distribution
        final_counts = y_balanced.value_counts()
        print(f"Balanced class distribution: {final_counts.to_dict()}")
        
        return X_balanced, y_balanced

    def fit(self, X, y):
        # Clean data to handle infinities
        X_clean = self._clean_data(X)
        
        # Additional validation for economic data
        if len(X_clean) != len(y):
            print(f"Warning: X and y length mismatch after cleaning. X: {len(X_clean)}, y: {len(y)}")
            # Align y with cleaned X
            y = y.loc[X_clean.index]
        
        # Check for class balance
        class_counts = y.value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        # Balance classes if needed (but not for extremely imbalanced data)
        min_class_ratio = min(class_counts) / len(y)
        if min_class_ratio >= 0.1 and self.balance_classes:  # Only balance if not extremely imbalanced and enabled
            # Apply class balancing
            X_clean, y = self._balance_classes(X_clean, y, method=self.balance_method, random_state=self.random_state)
        

        def objective(trial):
            # Simpler hyperparameter space for small datasets
            
            # Network architecture - smaller and simpler
            n_layers = trial.suggest_int('n_layers', 1, 3)  # Max 3 layers
            hidden_dims = []
            for i in range(n_layers):
                dim = trial.suggest_int(f'hidden_dim_{i}', 1, 5)
                hidden_dims.append(dim)
            
            # Training parameters optimized for small datasets
            lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])  # Smaller batch sizes
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.9)  # Higher dropout for regularization
            activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu', 'leaky_relu', 'selu'])
            #activation = trial.suggest_categorical('activation', ['tanh'])
            weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)  # Stronger regularization
            max_epochs = trial.suggest_int('max_epochs', 2, 30, log=True)  # Fewer epochs to prevent overfitting
            
            # Feature selection - more aggressive for small datasets
            use_feature_selection = trial.suggest_categorical('use_feature_selection', [True])
            k_features = None
            if use_feature_selection:
                # More aggressive feature selection for small datasets
                max_features = X_clean.shape[1]  # Use all features
                k_features = trial.suggest_int('k_features', 1, max_features)
            
            # Disable batch norm for small datasets
            use_batch_norm = False
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            
            # Get all the splits but only use the last two
            all_splits = list(tscv.split(X_clean))
            for train_idx, val_idx in all_splits[-2:]:
                
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

               # print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
                
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Feature selection if enabled
                if use_feature_selection and k_features is not None:
                    selector = SelectKBest(f_classif, k=k_features)
                    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                    X_val_selected = selector.transform(X_val_scaled)
                else:
                    X_train_selected = X_train_scaled
                    X_val_selected = X_val_scaled

                
                # Create PyTorch model with skorch wrapper
                input_dim = X_train_selected.shape[1]
                
                # Adjust batch size to training set size
                effective_batch_size = min(batch_size, len(X_train) // 3, 32)
                if effective_batch_size < 4:
                    effective_batch_size = min(16, len(X_train) // 2)
                
                net = NeuralNetClassifier(
                    MLPNet,
                    module__input_dim=input_dim,
                    module__hidden_dims=hidden_dims,
                    module__dropout_rate=dropout_rate,
                    module__activation=activation,
                    module__use_batch_norm=use_batch_norm,
                    max_epochs=max_epochs,
                    lr=lr,
                    batch_size=effective_batch_size,
                    optimizer=torch.optim.Adam,
                    optimizer__weight_decay=weight_decay,
                    criterion=nn.CrossEntropyLoss,
                    train_split=None,
                    verbose=0,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # Fit the model
                net.fit(X_train_selected.astype(np.float32), y_train.values.astype(np.int64))
                
                # Predict
                preds = net.predict(X_val_selected.astype(np.float32))
                score = accuracy_score(y_val, preds)
                scores.append(score)
                      # Skip this fol
            
            # Calculate Sharpe ratio as risk-adjusted performance metric
            if len(scores) == 0:
                return 1.0  # Return worst score if no successful folds
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Calculate Sharpe ratio (mean/std)
            if std_score == 0:
                # If all scores are identical, use mean as Sharpe ratio
                sharpe_ratio = mean_score
            else:
                sharpe_ratio = mean_score / std_score
            
            print(f"CV scores: {scores}, Mean: {mean_score:.4f}, Std: {std_score:.4f}, Sharpe: {sharpe_ratio:.4f}")
            if mean_score < 0.5:
                sharpe_ratio = sharpe_ratio - 5  # Penalize low performance
            return -mean_score  # Return negative Sharpe ratio to minimize

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        print(f"Best PyTorch Neural Network parameters: {self.best_params}")
        
        # Train the final model with best parameters
        try:
            self._train_final_model(X_clean, y)
            
        except Exception as e:
            print(f"PyTorch Neural Network training failed with error: {e}")
    
    
    def _train_final_model(self, X_clean, y):
        """Train the final PyTorch model with best parameters"""
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Feature selection if enabled
        if self.best_params.get('use_feature_selection', False) and 'k_features' in self.best_params:
            self.feature_selector = SelectKBest(f_classif, k=self.best_params['k_features'])
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            self.feature_selector = None
            X_selected = X_scaled
        
        # Build hidden dimensions from best params
        n_layers = self.best_params['n_layers']
        #n_layers = 1
        hidden_dims = [self.best_params[f'hidden_dim_{i}'] for i in range(n_layers)]
        
        # Create final model
        input_dim = X_selected.shape[1]
        
        # Determine batch size
        final_batch_size = self.best_params['batch_size']
        
        self.model = NeuralNetClassifier(
            MLPNet,
            module__input_dim=input_dim,
            module__hidden_dims=hidden_dims,
            module__dropout_rate=self.best_params['dropout_rate'],
            module__activation=self.best_params['activation'],
            module__use_batch_norm=False,  # Disabled for small datasets
            max_epochs=self.best_params['max_epochs'],
            lr=self.best_params['lr'],
            batch_size=final_batch_size,
            optimizer=torch.optim.Adam,
            optimizer__weight_decay=self.best_params['weight_decay'],
            criterion=nn.CrossEntropyLoss,
            train_split=None,
            verbose=0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Fit the model
        self.model.fit(X_selected.astype(np.float32), y.values.astype(np.int64))
        self.fitted = True
        
        # Evaluate on training data
        preds = self.model.predict(X_selected.astype(np.float32))
        print("Classification Report:\n", classification_report(y, preds))
        
        print(f"\nNeural Network Architecture (optimized for small datasets):")
        print(f"Input features: {input_dim}")
        print(f"Hidden layers: {hidden_dims}")
        print(f"Activation: {self.best_params['activation']}")
        print(f"Dropout rate: {self.best_params['dropout_rate']:.3f}")
        print(f"Learning rate: {self.best_params['lr']:.6f}")
        print(f"Weight decay: {self.best_params['weight_decay']:.6f}")
        print(f"Batch size: {final_batch_size}")
        print(f"Max epochs: {self.best_params['max_epochs']}")
        print(f"Feature selection: {self.best_params.get('use_feature_selection', False)}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            

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
        self.scaler = None
        self.feature_selector = None
        self.fit(X, y)
        
        # X_pred should be the same features as X
        X_pred = current_data.drop(columns=columns_to_drop)
        
        # Clean prediction data
        X_pred_clean = self._clean_data(X_pred)
        
        # Scale prediction data
        X_pred_scaled = self.scaler.transform(X_pred_clean)
        
        # Apply feature selection if used
        if self.feature_selector is not None:
            X_pred_selected = self.feature_selector.transform(X_pred_scaled)
        else:
            X_pred_selected = X_pred_scaled
        
        # Predict
        if 'fallback' in self.best_params:
            # Using fallback model
            pred = self.model.predict(X_pred_selected)[0]
            pred_probs = self.model.predict_proba(X_pred_selected)[0]
            print(f"Fallback model prediction probabilities: Class 0: {pred_probs[0]:.4f}, Class 1: {pred_probs[1]:.4f}")
        else:
            # Using PyTorch model
            pred = self.model.predict(X_pred_selected.astype(np.float32))[0]
            pred_probs = self.model.predict_proba(X_pred_selected.astype(np.float32))[0]
            print(f"PyTorch Neural Network prediction probabilities: Class 0: {pred_probs[0]:.4f}, Class 1: {pred_probs[1]:.4f}")
        
        return int(pred), 10  # signal, amount
