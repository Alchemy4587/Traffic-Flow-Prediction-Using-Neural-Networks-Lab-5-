# Traffic Flow Prediction Using Neural Networks

This project is designed to predict traffic flow using machine learning models, with an emphasis on deep learning techniques like LSTM. The project demonstrates data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization of results.

## Features

- **Data Loading & Exploration:** Reads traffic data and performs basic exploration to understand dataset characteristics.
- **Data Preprocessing:** Applies scaling techniques like MinMaxScaler or StandardScaler.
- **Model Training:** Implements an LSTM-based deep learning model.
- **Evaluation Metrics:** Uses Mean Absolute Error (MAE) and Mean Squared Error (MSE) for model performance evaluation.
- **Visualization:** Provides loss and metric plots and compares actual vs. predicted traffic flow.

## Requirements

To run this project, install the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Project Structure

- `main.py`: The main script containing the code for data loading, preprocessing, model building, training, and evaluation.

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Alchemy4587/Traffic-Flow-Prediction-Using-Neural-Networks-Lab-5-
   ```

2. **Prepare the Dataset:**
   Place your dataset file in the project directory (in the dataset folder). Ensure the dataset is in CSV format.

3. **Run the Script:**
   Execute the main script:
   ```bash
   python main.py
   ```

4. **Outputs:**
   - Statistical display of results
   - Preprocessing results and generated csv files that will be used in further steps
   - Training and validation loss and MAE plots.
   - Actual vs. predicted traffic flow plot.

## Key Functions

### `load_and_explore_data(file_path)`
Loads and displays basic information about the dataset, including data types, missing values, and a preview of the first few rows.

### `preprocess_data(df)`
Preprocesses the data by scaling and splitting it into training and validation datasets.

### `build_lstm_model()`
Defines and compiles an LSTM-based model with dropout layers for regularization.

### `train_and_evaluate_model(model, x_train, y_train, x_val, y_val)`
Trains the model using early stopping and evaluates its performance on the validation set.

### `plot_results(history)`
Plots training and validation loss and MAE over epochs.

### `visualize_predictions(y_true, y_pred)`
Creates a scatter plot to compare actual vs. predicted traffic flow.

## Sample Dataset
The project assumes a dataset with the following structure:

| DateTime          | Traffic_Flow |
|-------------------|--------------|
| 2025-01-01 00:00 | 1200         |
| 2025-01-01 01:00 | 1180         |

Make sure your dataset has a timestamp column and a target column for traffic flow.


Feel free to contribute by opening issues or submitting pull requests!
