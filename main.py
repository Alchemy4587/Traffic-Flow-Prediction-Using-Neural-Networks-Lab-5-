import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_and_explore_data(file_path):
    """
    Load and perform initial exploration of the traffic dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Basic information about the dataset
    print("\nDataset Info:")
    print(df.info())

    # Display first few rows
    print("\nFirst few rows of the dataset:")
    print(df.head())

    # Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Basic statistics of numerical columns
    print("\nBasic statistics of numerical columns:")
    print(df.describe())

    return df

#STEP 1 IMPLEMENTATION
def handle_time_series(df):
    """
    Process time-related features and sort data chronologically
    """
    # Create a copy of the dataframe to avoid warnings
    df = df.copy()

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Define rush hours (7-9 AM and 4-6 PM)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)

    # Sort data chronologically
    df = df.sort_values('timestamp')

    return df

def process_weather_data(df):
    """
    Process weather-related features and handle any missing values
    """
    # Create a copy of the dataframe to avoid warnings
    df = df.copy()

    # List of weather columns (modify based on your actual columns)
    weather_columns = ['temperature', 'humidity', 'precipitation']

    # Handle missing values in weather data
    for column in weather_columns:
        if column in df.columns:
            # Fill missing values with the mean of that column (avoiding inplace)
            df[column] = df[column].fillna(df[column].mean())

            # Handle outliers using IQR method
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap outliers at bounds
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df

def visualize_data(df):
    """
    Create visualizations to understand the data distribution and patterns
    """
    # Set figure style without using seaborn style
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['axes.grid'] = True

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2)

    # Traffic flow distribution
    axes[0, 0].hist(df['traffic_flow'], bins=30)
    axes[0, 0].set_title('Distribution of Traffic Flow')
    axes[0, 0].set_xlabel('Traffic Flow')
    axes[0, 0].set_ylabel('Frequency')

    # Traffic flow by hour
    hourly_traffic = df.groupby('hour')['traffic_flow'].mean()
    axes[0, 1].plot(hourly_traffic.index, hourly_traffic.values)
    axes[0, 1].set_title('Average Traffic Flow by Hour')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Average Traffic Flow')

    # Traffic flow by day of week
    daily_traffic = df.groupby('day_of_week')['traffic_flow'].mean()
    axes[1, 0].bar(daily_traffic.index, daily_traffic.values)
    axes[1, 0].set_title('Average Traffic Flow by Day of Week')
    axes[1, 0].set_xlabel('Day of Week (0=Monday)')
    axes[1, 0].set_ylabel('Average Traffic Flow')

    # If weather data is available, plot correlation with traffic
    if 'temperature' in df.columns:
        axes[1, 1].scatter(df['temperature'], df['traffic_flow'], alpha=0.5)
        axes[1, 1].set_title('Traffic Flow vs Temperature')
        axes[1, 1].set_xlabel('Temperature')
        axes[1, 1].set_ylabel('Traffic Flow')

    plt.tight_layout()
    plt.show()

    return fig

#STEP 2 IMPLEMENTATION
class TrafficFeatureEngineering:
    def __init__(self, df):
        """
        Initialize the feature engineering class with the dataframe
        """
        self.df = df.copy()
        self.scalers = {}

    def create_time_features(self):
        """
        Create advanced time-based features
        """
        # Extract basic time features
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)

        # Day of week features (cyclical encoding)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)

        # Rush hour periods (morning: 7-9, evening: 16-18)
        self.df['is_morning_rush'] = self.df['hour'].isin([7, 8, 9]).astype(int)
        self.df['is_evening_rush'] = self.df['hour'].isin([16, 17, 18]).astype(int)

        # Part of day
        self.df['part_of_day'] = pd.cut(self.df['hour'],
                                        bins=[-1, 6, 12, 18, 23],
                                        labels=['night', 'morning', 'afternoon', 'evening'])

        # One-hot encode part of day
        part_of_day_dummies = pd.get_dummies(self.df['part_of_day'], prefix='time')
        self.df = pd.concat([self.df, part_of_day_dummies], axis=1)

        return self.df

    def process_weather_features(self):
        """
        Process and normalize weather-related features
        """
        weather_features = ['temperature', 'humidity', 'precipitation']

        for feature in weather_features:
            if feature in self.df.columns:
                # Create interaction features with time
                self.df[f'{feature}_rush_hour'] = (
                        self.df[feature] * (self.df['is_morning_rush'] | self.df['is_evening_rush'])
                )

                # Normalize weather features
                scaler = StandardScaler()
                self.df[f'{feature}_normalized'] = scaler.fit_transform(
                    self.df[feature].values.reshape(-1, 1)
                )
                self.scalers[feature] = scaler

        return self.df

    def create_lag_features(self, target_col='traffic_flow', lags=[1, 2, 3, 24]):
        """
        Create lag features for the target column
        """
        print(f"Shape before creating lag features: {self.df.shape}")  # Debug print

        for lag in lags:
            # Create lag feature
            self.df[f'{target_col}_lag_{lag}'] = self.df[target_col].shift(lag)

            # Create rolling mean features
            self.df[f'{target_col}_rolling_mean_{lag}'] = (
                self.df[target_col].rolling(window=lag, min_periods=1).mean()  # Added min_periods=1
            )

            # Create rolling std features
            self.df[f'{target_col}_rolling_std_{lag}'] = (
                self.df[target_col].rolling(window=lag, min_periods=1).std()  # Added min_periods=1
            )

        # Instead of dropping all NaN values, only drop rows where all lag features are NaN
        lag_columns = [col for col in self.df.columns if 'lag' in col or 'rolling' in col]
        self.df = self.df.dropna(subset=lag_columns, how='all')

        print(f"Shape after creating lag features: {self.df.shape}")  # Debug print
        return self.df

    def normalize_features(self, method='standard'):
        """
        Normalize numerical features using specified method
        """
        print(f"Shape before normalization: {self.df.shape}")  # Debug print

        # Features to normalize
        numerical_features = [
            col for col in self.df.columns
            if self.df[col].dtype in ['int64', 'float64']
               and col != 'timestamp'  # Exclude timestamp
               and not self.df[col].isnull().all()  # Exclude columns that are all NaN
        ]

        print(f"Number of numerical features to normalize: {len(numerical_features)}")  # Debug print

        if len(numerical_features) == 0:
            print("Warning: No numerical features found for normalization")
            return self.df

        # Choose scaler based on method
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()

        # Fit and transform the selected features
        try:
            self.df[numerical_features] = scaler.fit_transform(self.df[numerical_features])
            self.scalers['features'] = scaler
        except Exception as e:
            print(f"Error during normalization: {e}")
            print(f"DataFrame shape: {self.df.shape}")
            print(f"Numerical features: {numerical_features}")
            raise

        print(f"Shape after normalization: {self.df.shape}")  # Debug print
        return self.df

    def get_feature_importance(self, target_col='traffic_flow'):
        """
        Calculate correlation between features and target
        """
        # Select only numeric columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        # Calculate correlations using only numeric columns
        correlations = self.df[numeric_cols].corr()[target_col].sort_values(ascending=False)
        return correlations

# STEP 3 AND 4 IMPLEMENTATION
class TrafficPredictionModel:
    def __init__(self, df, target_col='traffic_flow'):
        """
        Initialize the model class
        """
        self.df = df
        self.target_col = target_col
        self.model = None
        self.history = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def prepare_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into train, validation, and test sets
        """
        # Separate features and target
        X = self.df.drop([self.target_col, 'timestamp', 'part_of_day'], axis=1)
        y = self.df[self.target_col]

        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Second split: separate validation set from training set
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )

        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Test set shape: {self.X_test.shape}")

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def build_fcn_model(self, input_dim):
        """
        Build a Fully Connected Neural Network
        """
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),

            # Hidden layers
            Dense(64, activation='relu'),
            Dropout(0.2),

            Dense(32, activation='relu'),
            Dropout(0.2),

            # Output layer
            Dense(1)  # Linear activation for regression
        ])

        return model

    def build_lstm_model(self, input_dim):
        """
        Build an LSTM model for time series prediction
        """
        model = Sequential([
            # LSTM layers
            LSTM(128, return_sequences=True, input_shape=(input_dim, 1)),
            Dropout(0.3),

            LSTM(64, return_sequences=False),
            Dropout(0.2),

            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.2),

            # Output layer
            Dense(1)  # Linear activation for regression
        ])

        return model

    def train_model(self, model_type='fcn', epochs=100, batch_size=32):
        """
        Train the selected model type
        """
        input_dim = self.X_train.shape[1]

        # Build the selected model type
        if model_type.lower() == 'lstm':
            # Reshape data for LSTM
            self.X_train = self.X_train.values.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
            self.X_val = self.X_val.values.reshape((self.X_val.shape[0], self.X_val.shape[1], 1))
            self.X_test = self.X_test.values.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
            self.model = self.build_lstm_model(input_dim)
        else:
            self.model = self.build_fcn_model(input_dim)

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )

        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        return self.model, self.history

    def evaluate_model(self):
        """
        Evaluate the model on test set
        """
        test_loss, test_mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"\nTest Loss (MSE): {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")

        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Calculate additional metrics
        mse = tf.keras.metrics.mean_squared_error(self.y_test, y_pred).numpy().mean()
        rmse = np.sqrt(mse)

        print(f"Root Mean Squared Error: {rmse:.4f}")

        return mse, rmse, y_pred

#STEP 5 AND 6 IMPLEMENTATION
class ModelTrainingEvaluation:
    def __init__(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.history = None

    def train_model(self, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train the model with early stopping and learning rate reduction
        """
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return self.history

    def evaluate_model(self):
        """
        Evaluate model performance using multiple metrics
        """
        # Get predictions and ensure they're 1D
        y_pred = self.model.predict(self.X_test)
        y_pred = y_pred.flatten()  # Convert predictions to 1D array

        # Ensure y_test is also 1D
        y_test_values = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        y_test_values = y_test_values.flatten()

        # Calculate metrics
        mse = mean_squared_error(y_test_values, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_values, y_pred)

        # Calculate R-squared
        ss_res = np.sum((y_test_values - y_pred) ** 2)
        ss_tot = np.sum((y_test_values - np.mean(y_test_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print("\nModel Performance Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")

        return mse, rmse, mae, r2, y_pred

    def plot_training_history(self):
        """
        Plot training history to visualize learning progress
        """
        plt.figure(figsize=(12, 4))

        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training & validation MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, y_pred):
        """
        Plot actual vs predicted values
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'r--', lw=2)
        plt.xlabel('Actual Traffic Flow')
        plt.ylabel('Predicted Traffic Flow')
        plt.title('Actual vs Predicted Traffic Flow')
        plt.tight_layout()
        plt.show()


class ModelTuning:
    def __init__(self, X, y):
        self.X = np.asarray(X.values if hasattr(X, 'values') else X, dtype=np.float32)
        self.y = np.asarray(y.values if hasattr(y, 'values') else y, dtype=np.float32)

    def build_model(self, input_shape):
        """
        Build model using proper input shape
        """
        model = Sequential([
            # Use Input layer as first layer
            Input(shape=(input_shape,)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        return model

    def cross_validate_model(self, model_builder, n_splits=3):  # Reduced from 5 to 3
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            model = model_builder(X_train.shape[1])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

            # Reduced epochs from 50 to 20
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=32,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                verbose=1
            )

            val_mse = history.history['val_loss'][-1]
            fold_scores.append(np.sqrt(val_mse))

        return fold_scores

    def grid_search_hyperparameters(self, X_train, X_val, y_train, y_val):
        # Reduced parameter combinations
        param_grid = {
            'learning_rates': [0.01, 0.001],  # Removed 0.1
            'batch_sizes': [32, 64],  # Removed 16
            'hidden_layers': [
                [128, 64, 32],
                [64, 32, 16]
            ]  # Removed one combination
        }

        best_val_loss = float('inf')
        best_params = None

        for lr in param_grid['learning_rates']:
            for bs in param_grid['batch_sizes']:
                for hidden in param_grid['hidden_layers']:
                    model = Sequential([
                        Input(shape=(X_train.shape[1],)),
                        Dense(hidden[0], activation='relu'),
                        Dropout(0.3),
                        Dense(hidden[1], activation='relu'),
                        Dropout(0.2),
                        Dense(hidden[2], activation='relu'),
                        Dropout(0.2),
                        Dense(1)
                    ])

                    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

                    # Reduced epochs from 50 to 20
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20,
                        batch_size=bs,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                        verbose=1
                    )

                    val_loss = min(history.history['val_loss'])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {
                            'learning_rate': lr,
                            'batch_size': bs,
                            'hidden_layers': hidden
                        }

        return best_params, None
        # def cross_validate_model(self, model_builder, n_splits=5): #Changed from 5 to 3 because it took too much time to run
    #     """
    #     Perform k-fold cross-validation
    #     """
    #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    #     fold_scores = []
    #
    #     print(f"\nPerforming {n_splits}-fold cross-validation...")
    #
    #     for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
    #         print(f"\nFold {fold + 1}/{n_splits}")
    #
    #         # Split data using numpy array indexing
    #         X_train = self.X[train_idx]
    #         X_val = self.X[val_idx]
    #         y_train = self.y[train_idx]
    #         y_val = self.y[val_idx]
    #
    #         # Build model using proper input shape
    #         model = self.build_model(X_train.shape[1])
    #
    #         model.compile(
    #             optimizer=Adam(learning_rate=0.001),
    #             loss='mse',
    #             metrics=['mae']
    #         )
    #
    #         # Add early stopping
    #         early_stopping = EarlyStopping(
    #             monitor='val_loss',
    #             patience=5,
    #             restore_best_weights=True
    #         )
    #
    #         # Train model
    #         try:
    #             history = model.fit(
    #                 X_train, y_train,
    #                 validation_data=(X_val, y_val),
    #                 epochs=50,
    #                 batch_size=32,
    #                 callbacks=[early_stopping],
    #                 verbose=1
    #             )
    #
    #             # Evaluate model
    #             val_mse = history.history['val_loss'][-1]
    #             fold_scores.append(np.sqrt(val_mse))  # RMSE
    #             print(f"Fold {fold + 1} RMSE: {np.sqrt(val_mse):.4f}")
    #
    #         except Exception as e:
    #             print(f"Error during training fold {fold + 1}: {str(e)}")
    #             continue
    #
    #     if fold_scores:
    #         print("\nCross-validation results:")
    #         print(f"Mean RMSE: {np.mean(fold_scores):.4f}")
    #         print(f"Std RMSE: {np.std(fold_scores):.4f}")
    #     else:
    #         print("\nNo successful folds completed")
    #
    #     return fold_scores
    #
    # def grid_search_hyperparameters(self, X_train, X_val, y_train, y_val):
    #     """
    #     Perform grid search for hyperparameter tuning
    #     """
    #     # Convert inputs to numpy arrays and ensure float32 type
    #     X_train = np.asarray(X_train.values if hasattr(X_train, 'values') else X_train, dtype=np.float32)
    #     X_val = np.asarray(X_val.values if hasattr(X_val, 'values') else X_val, dtype=np.float32)
    #     y_train = np.asarray(y_train.values if hasattr(y_train, 'values') else y_train, dtype=np.float32)
    #     y_val = np.asarray(y_val.values if hasattr(y_val, 'values') else y_val, dtype=np.float32)
    #
    #     param_grid = {
    #         'learning_rates': [0.1, 0.01, 0.001],
    #         'batch_sizes': [16, 32, 64],
    #         'hidden_layers': [
    #             [128, 64, 32],
    #             [256, 128, 64],
    #             [64, 32, 16]
    #         ]
    #     }
    #
    #     best_val_loss = float('inf')
    #     best_params = None
    #     results = []
    #
    #     print("\nPerforming grid search for hyperparameters...")
    #
    #     for lr in param_grid['learning_rates']:
    #         for bs in param_grid['batch_sizes']:
    #             for hidden in param_grid['hidden_layers']:
    #                 print(f"\nTrying: lr={lr}, batch_size={bs}, hidden_layers={hidden}")
    #
    #                 # Build model with input layer
    #                 model = Sequential([
    #                     Input(shape=(X_train.shape[1],)),
    #                     Dense(hidden[0], activation='relu'),
    #                     Dropout(0.3),
    #                     Dense(hidden[1], activation='relu'),
    #                     Dropout(0.2),
    #                     Dense(hidden[2], activation='relu'),
    #                     Dropout(0.2),
    #                     Dense(1)
    #                 ])
    #
    #                 model.compile(
    #                     optimizer=Adam(learning_rate=lr),
    #                     loss='mse',
    #                     metrics=['mae']
    #                 )
    #
    #                 early_stopping = EarlyStopping(
    #                     monitor='val_loss',
    #                     patience=5,
    #                     restore_best_weights=True
    #                 )
    #
    #                 try:
    #                     history = model.fit(
    #                         X_train, y_train,
    #                         validation_data=(X_val, y_val),
    #                         epochs=50,
    #                         batch_size=bs,
    #                         callbacks=[early_stopping],
    #                         verbose=1
    #                     )
    #
    #                     val_loss = min(history.history['val_loss'])
    #                     results.append({
    #                         'learning_rate': lr,
    #                         'batch_size': bs,
    #                         'hidden_layers': hidden,
    #                         'val_loss': val_loss
    #                     })
    #
    #                     if val_loss < best_val_loss:
    #                         best_val_loss = val_loss
    #                         best_params = {
    #                             'learning_rate': lr,
    #                             'batch_size': bs,
    #                             'hidden_layers': hidden
    #                         }
    #
    #                     print(f"Validation loss: {val_loss:.4f}")
    #
    #                 except Exception as e:
    #                     print(f"Error during grid search iteration: {str(e)}")
    #                     continue
    #
    #     if best_params:
    #         print("\nBest hyperparameters found:")
    #         print(f"Learning rate: {best_params['learning_rate']}")
    #         print(f"Batch size: {best_params['batch_size']}")
    #         print(f"Hidden layers: {best_params['hidden_layers']}")
    #         print(f"Best validation loss: {best_val_loss:.4f}")
    #     else:
    #         print("\nNo successful hyperparameter combinations found")
    #
    #     return best_params, results

def predict_future_traffic(model, last_known_data, future_features, n_steps=24):
    """
    Predict traffic flow for future time periods
    """
    predictions = []

    # Convert input to numeric and normalize
    last_known_data = pd.to_numeric(last_known_data, errors='coerce').fillna(0)
    current_input = pd.to_numeric(last_known_data, errors='coerce').to_numpy().reshape(1, -1)  # Convert to numpy array properly

    for _ in range(n_steps):
        # Make prediction for next step
        next_pred = model.predict(current_input, verbose=0)[0][0]
        predictions.append(next_pred)

        # Update input for next prediction
        # This assumes the features are properly organized for the next time step
        current_input = np.roll(current_input, -1)
        current_input[0, -1] = next_pred

    return predictions

def visualize_predictions(y_actual, y_predicted, timestamps):
    """
    Plot the actual vs predicted traffic flow over time.
    """
    plt.figure(figsize=(12, 6))

    # Plot the actual vs predicted traffic flow
    plt.plot(timestamps, y_actual, label="Actual Traffic Flow", color="blue")
    plt.plot(timestamps, y_predicted, label="Predicted Traffic Flow", color="orange", linestyle="--")

    plt.title("Traffic Flow: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Traffic Flow")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#STEP 7
def visualize_weather_vs_traffic(df):
    """
    Create scatter plots to visualize the relationship between weather conditions and traffic flow.
    """
    plt.figure(figsize=(18, 6))

    # Temperature vs Traffic Flow
    if 'temperature' in df.columns:
        plt.subplot(1, 3, 1)
        plt.scatter(df['temperature'], df['traffic_flow'], alpha=0.5, color='red')
        plt.title("Traffic Flow vs Temperature")
        plt.xlabel("Temperature")
        plt.ylabel("Traffic Flow")
        plt.grid(True)

    # Humidity vs Traffic Flow
    if 'humidity' in df.columns:
        plt.subplot(1, 3, 2)
        plt.scatter(df['humidity'], df['traffic_flow'], alpha=0.5, color='green')
        plt.title("Traffic Flow vs Humidity")
        plt.xlabel("Humidity")
        plt.ylabel("Traffic Flow")
        plt.grid(True)

    # Precipitation vs Traffic Flow
    if 'precipitation' in df.columns:
        plt.subplot(1, 3, 3)
        plt.scatter(df['precipitation'], df['traffic_flow'], alpha=0.5, color='blue')
        plt.title("Traffic Flow vs Precipitation")
        plt.xlabel("Precipitation")
        plt.ylabel("Traffic Flow")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # File path
    file_path = "dataset/synthetic_traffic_data - synthetic_traffic_data.csv"

    # Step 1: Load and explore the dataset
    print("Loading and exploring the dataset...")
    df = load_and_explore_data(file_path)

    # Step 2: Handle time series data
    print("\nProcessing time-series data...")
    df = handle_time_series(df)

    # Step 3: Process weather data
    print("\nProcessing weather data...")
    df = process_weather_data(df)

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_data(df)

    # Save processed dataset
    processed_file_path = "dataset/processed_traffic_data.csv"
    df.to_csv(processed_file_path, index=False)
    print(f"\nProcessed data saved to: {processed_file_path}")

    # Feature Engineering
    print("\nStarting feature engineering...")
    df = pd.read_csv('dataset/processed_traffic_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Initialize feature engineering class
    fe = TrafficFeatureEngineering(df)
    print(f"Initial DataFrame shape: {df.shape}")

    # Apply feature engineering steps
    print("Creating time features...")
    df = fe.create_time_features()
    print(f"Shape after time features: {df.shape}")

    print("Processing weather features...")
    df = fe.process_weather_features()
    print(f"Shape after weather features: {df.shape}")

    print("Creating lag features...")
    df = fe.create_lag_features(lags=[1, 2, 3, 24])
    print(f"Shape after lag features: {df.shape}")

    print("Normalizing features...")
    df = fe.normalize_features(method='standard')
    print(f"Final shape: {df.shape}")

    # Save engineered features
    output_path = 'dataset/engineered_traffic_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nEngineered features saved to: {output_path}")

    # Model Training
    print("\nStarting model training...")
    # Load the engineered data
    df = pd.read_csv('dataset/engineered_traffic_data.csv')

    # Initialize model class
    print("Initializing model...")
    traffic_model = TrafficPredictionModel(df)

    # Prepare data
    print("Preparing data splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = traffic_model.prepare_data()

    # Build the model first - this is the key fix
    input_dim = X_train.shape[1]
    model = traffic_model.build_fcn_model(input_dim)

    # Initialize model training and evaluation with the built model
    model_trainer = ModelTrainingEvaluation(
        model,  # Pass the built model instead of traffic_model.model
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )

    # Train the model
    # print("\nTraining model...")
    # history = model_trainer.train_model(epochs=100, batch_size=32)

    print("\nTraining model...")
    history = model_trainer.train_model(epochs=50, batch_size=32)

    # Evaluate and visualize results
    print("\nEvaluating model...")
    mse, rmse, mae, r2, predictions = model_trainer.evaluate_model()

    # Plot training history
    model_trainer.plot_training_history()

    # Plot predictions
    model_trainer.plot_predictions(predictions)

    print("\nPerforming model tuning...")
    # Ensure data is numeric and handle any non-numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_cols]
    X_val_numeric = X_val[numeric_cols]

    model_tuner = ModelTuning(X_train_numeric, y_train)

    # Cross-validation
    cv_scores = model_tuner.cross_validate_model(
        model_builder=traffic_model.build_fcn_model
    )

    # Grid search for hyperparameters
    best_params, tuning_results = model_tuner.grid_search_hyperparameters(
        X_train_numeric, X_val_numeric,
        y_train, y_val
    )

    # Predict future traffic (example)
    print("\nPredicting future traffic...")
    last_known_data = X_test.iloc[-1]  # Use iloc instead of direct indexing
    future_predictions = predict_future_traffic(
        model,  # Use the trained model
        last_known_data,
        None,
        n_steps=24
    )

    return model, history, predictions, best_params, future_predictions

    # Visualising weather vs traffic
    timestamps = X_test['timestamp']  # Ensure 'timestamp' is included in X_test
    visualize_predictions(y_test, predictions, timestamps)

    # Visualizing weather vs traffic
    visualize_weather_vs_traffic(df)  # Use the original dataset `df` after merging and preprocessing


if __name__ == "__main__":
    # try:
    #     model, history, predictions, best_params, future_preds = main()
    #
    #     print("Training completed successfully!")
    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")
    model, history, predictions, best_params, future_preds = main()