import pandas as pd
import numpy as np
import optuna


import pytorch_lightning as pl

from pytorch_forecasting.metrics import QuantileLoss, RMSE, MAE, MAPE
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from strategy import Strategy

import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import pandas as pd
#from pandas.core.common import SettingWithCopyWarning
import torch

from pytorch_forecasting import CrossEntropy, GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.models.deepar import DeepAR

torch.set_float32_matmul_precision('medium')


class TemporalFusionTransformerOptunaStrategy(Strategy):
    def __init__(self, n_trials=3, n_splits=3, random_state=42, prediction_length=1, max_encoder_length=5):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.fitted = False

        # Temporal Fusion Transformer specific parameters
        self.prediction_length = prediction_length  # Number of future time steps to predict
        self.max_encoder_length = max_encoder_length  # Max history length
        self.trainer = None
        self.training_data = None
        self.validation_data = None
        self.scaler = StandardScaler()  # For feature normalization
        
    def _clean_data(self, df):
        """Clean data by removing rows with NaN or infinity values"""
        # Make a copy to avoid modifying the original data
        df_clean = df.copy()
        
        # Remove rows with infinity values
        df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
        
        # Remove rows with NaN values
        df_clean = df_clean.dropna()
        
        return df_clean
    
    def _prepare_timeseries_data(self, df, is_train=True):
        """
        Prepare data for TimeSeriesDataSet format required by PyTorch Forecasting
        """
        # Add a time_idx column - this is required for TimeSeriesDataSet
        if 'Date' in df.columns:
            df = df.sort_values('Date')
            # Create a time index starting from 0
            df['time_idx'] = range(len(df))
        else:
            # If no Date column, create a time index based on the dataframe index
            df = df.reset_index(drop=True)
            df['time_idx'] = df.index
        
        # Add a group_id column - DeepAR requires this for identifying different time series
        # Since we have a single time series, we use a constant value
        df['group_id'] = 'forex'  # All rows belong to the same group
        
        # For training data, we need to set up both training and validation datasets
        if is_train:
            # Determine the cutoff point for validation
            cutoff = int(len(df) * 0.6)  # Use 60% for training

            # Normalize features
            feature_columns = [col for col in df.columns if col not in ['Date', 'Label', 'time_idx', 'group_id']]
            df_features = df[feature_columns]
            
            # Fit scaler on training data only to avoid data leakage
            train_features = df_features.iloc[:cutoff]
            self.scaler.fit(train_features)
            
            # Transform all features
            normalized_features = self.scaler.transform(df_features)
            for i, col in enumerate(feature_columns):
                df[col] = normalized_features[:, i]
            
            target = 'Label'
                # No need to normalize binary classification labels
            
            # Create training dataset
            self.training_data = TimeSeriesDataSet(
                data=df.iloc[:cutoff],
                time_idx="time_idx",
                target=target,
                group_ids=["group_id"],
                static_categoricals=["group_id"],
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=self.prediction_length,
                time_varying_unknown_reals=[target],
                time_varying_known_reals=[col for col in df.columns 
                                          if col not in ['Date', 'Label', target, 'time_idx', 'group_id']],
                target_normalizer=NaNLabelEncoder(),
                predict_mode=False
            )

            all_training_data = TimeSeriesDataSet(
                data=df,
                time_idx="time_idx",
                target=target,
                group_ids=["group_id"],
                static_categoricals=["group_id"],
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=self.prediction_length,
                time_varying_unknown_reals=[target],
                time_varying_known_reals=[col for col in df.columns 
                                          if col not in ['Date', 'Label', target, 'time_idx', 'group_id']],
                target_normalizer=NaNLabelEncoder(),
                predict_mode=False
            )

            all_training_data_pred = TimeSeriesDataSet(
                data=df,
                time_idx="time_idx",
                target=target,
                group_ids=["group_id"],
                static_categoricals=["group_id"],
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=self.prediction_length,
                time_varying_unknown_reals=[target],
                time_varying_known_reals=[col for col in df.columns 
                                          if col not in ['Date', 'Label', target, 'time_idx', 'group_id']],
                target_normalizer=NaNLabelEncoder(),
                predict_mode=False
            )
            
            # Create validation dataset with the same parameters
            self.validation_data = TimeSeriesDataSet.from_dataset(
                self.training_data, df.iloc[cutoff-self.max_encoder_length:], predict=True, stop_randomization=True
            )
            
            # Create data loaders
            train_dataloader = self.training_data.to_dataloader(
                batch_size=32, num_workers=0, shuffle=True
            )
            val_dataloader = self.validation_data.to_dataloader(
                batch_size=32, num_workers=0, shuffle=False
            )

            all_train_dataloader = all_training_data.to_dataloader(
                batch_size=32, num_workers=0, shuffle=True
            )

            all_train_dataloader_pred = all_training_data_pred.to_dataloader(
                batch_size=1, num_workers=0, shuffle=False
            )

            return train_dataloader, val_dataloader, df.iloc[self.max_encoder_length:], all_train_dataloader, all_train_dataloader_pred

        else:
            # For prediction data, we just normalize using the already fitted scalers
            feature_columns = [col for col in df.columns if col not in ['Date', 'Label', 'time_idx', 'group_id']]
            df_features = df[feature_columns]
            
            normalized_features = self.scaler.transform(df_features)
            for i, col in enumerate(feature_columns):
                df[col] = normalized_features[:, i]
            
            # Always prioritize using Label as the target, just like in training
            target = 'Label'
            
            # Create dataset for prediction
            prediction_dataset = TimeSeriesDataSet(
                data=df,
                time_idx="time_idx",
                target=target,
                group_ids=["group_id"],
                static_categoricals=["group_id"],
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=self.prediction_length,
                time_varying_unknown_reals=[target],
                time_varying_known_reals=[col for col in df.columns 
                                          if col not in ['Date', 'Label', target, 'time_idx', 'group_id']],
                target_normalizer=NaNLabelEncoder(),
                predict_mode=True
            )
            
            pred_dataloader = prediction_dataset.to_dataloader(
                batch_size=1, num_workers=0, shuffle=False
            )
            
            return pred_dataloader, target, None, None, None

    def fit(self, X, y):
        """
        Fit the DeepAR model on the training data
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series or np.array
            Target vector
        """
        # Clean data to handle infinities
        X_clean = self._clean_data(X)
        
        # Combine features and target for time series dataset creation
        data = X_clean.copy()
        data['Label'] = y

        print(f"Data Label Sample Last 5 Rows:\n{data['Label'].tail(20)}")
        
        # Create time series datasets
        train_dataloader, val_dataloader, train_df, all_train_dataloader, all_train_dataloader_pred = self._prepare_timeseries_data(data, is_train=True)
        
        if train_dataloader is None or val_dataloader is None:
            print("Error preparing time series data.")
            return
        
        def objective(trial):
            # Define the hyperparameter search space for DeepAR
            params = {
                'hidden_size': trial.suggest_int('hidden_size', 16, 100),
                #'rnn_layers': trial.suggest_int('rnn_layers', 5, 10),
                'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'max_epochs': trial.suggest_int('max_epochs', 20, 50, step=10),
            }
            
            # Define DeepAR model with the trial parameters
            model = TemporalFusionTransformer.from_dataset(
                self.training_data,
                learning_rate=params['learning_rate'],
                hidden_size=params['hidden_size'],
                #rnn_layers=params['rnn_layers'],
                dropout=params['dropout'],
                output_size=2,                # two classes (0 or 1):contentReference[oaicite:4]{index=4}
                loss=CrossEntropy()
            )
            
            # Debugging: Log the type of the model
            print(f"Model type: {type(model)}")
            
            
            trainer = pl.Trainer(
                max_epochs=params['max_epochs'],
                accelerator='gpu',  # Use GPU
                devices=1,  # Use one GPU
                gradient_clip_val=0.1,
                enable_progress_bar=False,  # Disable progress bar during optimization
                logger=False,  # Disable logging during optimization
            )
            
            # Train model
            try:
                trainer.fit(
                    model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                )
                
                # Get validation loss
                val_loss = trainer.callback_metrics.get("val_loss").item()
                return val_loss
                
            except Exception as e:
                print(f"Error in DeepAR training: {e}")
                return float('inf')  # Return worst score for failed trials
        
        # Run the optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get best parameters
        self.best_params = study.best_params
        print(f"Best DeepAR parameters: {self.best_params}")
        
        # Train the final model with best parameters
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_data,
            learning_rate=self.best_params['learning_rate'],
            hidden_size=self.best_params['hidden_size'],
            #rnn_layers=self.best_params['rnn_layers'],
            dropout=self.best_params['dropout'],
            output_size=2,                # two classes (0 or 1):contentReference[oaicite:4]{index=4}
            loss=CrossEntropy()
        )
        
        
        self.trainer = pl.Trainer(
            max_epochs=self.best_params['max_epochs'],
            accelerator='gpu',  # Use CPU
            gradient_clip_val=0.1,
            #devices=1,  # Use one GPU
        )
        
        # Train the final model
        self.trainer.fit(
            self.model,
            train_dataloaders=all_train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        self.fitted = True
        print("DeepAR model training completed.")
        # predict train_data_loader and then run classification report print("Classification Report:\n", classification_report(y, preds))
        # predictions = self.model.predict(all_train_dataloader_pred)
        # Predict on test set
        predictions = self.model.predict(all_train_dataloader_pred, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
       # print(predictions.output);
       # print(predictions.y);
        y_pred_test = predictions.output.numpy().flatten()
        y_true_test = predictions.y[0].numpy().flatten()
        #preds = predictions[:, -1]  # Get the last prediction for each sequence
        #preds = preds.cpu().numpy()  # Convert to numpy array
        print(f"Predictions Sample Last 5 Rows:\n{y_pred_test[-5:]}")
        # For Label target (classification), threshold the output to get the class
        # preds = (preds >= 0.0).astype(int)  # Convert probabilities to binary predictions
        #labels_int = train_df["Label"]
        print("Classification Report:\n", classification_report(y_true_test, y_pred_test))

        
        
    def generate_signal(self, past_data, current_data):
        """
        Generate trading signals using the DeepAR model
        
        Parameters:
        -----------
        past_data : pd.DataFrame
            Historical data for model training
        current_data : pd.DataFrame
            Current market data for prediction
        
        Returns:
        --------
        signal : int
            1 for buy signal, 0 for sell/hold
        amount : int
            Trading amount (fixed at 10)
        """
        # Clean input data
        past_data = self._clean_data(past_data)
        current_data = self._clean_data(current_data)

        if current_data.empty:
            return None, 10
        
        # Make sure we have enough data for the encoder
        if len(past_data) < self.max_encoder_length:
            print(f"Not enough past data for encoder. Need at least {self.max_encoder_length} rows.")
            return None, 10
        
        # Reset the model and normalizers every time we call generate_signal
        self.model = None
        self.trainer = None
        self.scaler = StandardScaler()  # For feature normalization
        
        # Extract features and target from past_data (consistent with other strategies)
        columns_to_drop = ['Label', 'Date']
        X = past_data.drop(columns=columns_to_drop)
        # Always use Label as target (y) for training
        y = past_data["Label"]

        # Fit the model on the current past data
        self.fit(X, y)
        
        # Combine past and current data for DeepAR prediction
        # We need the last max_encoder_length rows from past_data for the context window
        combined_data = pd.concat([past_data.iloc[-self.max_encoder_length:], current_data])
        
        # Prepare data for prediction
        pred_dataloader, target_name, pred_data, ignore, all_pred_dataloader = self._prepare_timeseries_data(combined_data, is_train=False)

        if pred_dataloader is None:
            return None, 10
        
        # Get predictions from the model
        predictions = self.model.predict(pred_dataloader, trainer_kwargs=dict(accelerator="cpu"))
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction content: {predictions}")
        # median_prediction = predictions[:, -1]  
        # For Label target (classification), threshold the output to get the class
        pred_value = predictions.numpy().flatten()[0]
        # signal = 1 if pred_value >= 0.0 else 0
        print(f"Signal: {'BUY' if pred_value == 1 else 'SELL'}")
        return int(pred_value), 10  # signal, amount
