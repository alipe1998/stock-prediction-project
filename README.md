# Stock Prediction Project

## Project Structure

```
stock-prediction-project/
│
├── README.md                         
├── requirements.txt                  
├── .env                              
│
├── data/
│   ├── raw/                          
│   ├── processed/                    
│   ├── live/                         
│   └── models/                      # Trained models stored here
│
├── notebooks/                        
│   ├── exploratory_analysis.ipynb    
│   ├── model_testing.ipynb          
│   └── backtesting.ipynb            # Notebook dedicated to backtesting
│
├── src/
│   ├── config/
│   │   ├── config.yaml               # MAIN configuration file
│   │   └── model_registry.py         # Dict of supported models and preprocessors
│   │
│   ├── data/
│   │   ├── fetch_data.py             
│   │   └── preprocess.py             
│   │
│   ├── models/
│   │   ├── base_model.py            # Abstract Base Class
│   │   ├── mlp.py                   # MLP-specific class
│   │   └── xgboost_model.py         # XGBoost example
│   │
│   ├── backtest/
│   │   ├── backtest.py               
│   │   └── evaluation.py             
│   │
│   ├── trading/
│   │   ├── alpaca_trade.py           
│   │   └── portfolio.py              
│   │
│   └── utils/
│       ├── helpers.py                
│       └── logger.py                 # Custom logging
│
└── reports/
    └── project_report.pdf            
```

### How the Backtest Works

Let's break down how your backtesting function works and what results you should expect.

---

## How the Backtesting Works

1. **Rolling Window Data Pooling:**  
   - **Training Period Setup:**  
     The code defines a series of "training endpoints" (months) using the date range between `start_month` and `end_month` (with monthly frequency).  
   - **Pooling Training Data:**  
     For each training endpoint (say, current month), the function gathers data from the past `months_back` months. It does this by:
       - Calculating a list of months (e.g., if `months_back` is 3 and the current month is April 2010, it pools data from April, March, and February 2010).
       - Selecting the tickers that are common across all these months.
       - Returning the features (X_train) and the target (y_train) for those tickers.

2. **Preprocessing:**  
   - Once the training data is pooled, it passes X_train through your preprocessing pipeline (which can include steps such as dropping NaNs, filtering by price, applying a quantile transformation, generating polynomial features, and encoding industry).
   - After preprocessing, y_train is re-aligned to match the indices of X_train. This is important because some preprocessing steps might drop rows.

3. **Model Training via Grid Search:**  
   - The preprocessed training data is then used in a grid search to tune hyperparameters for your model (like MLP or XGBoost).  
   - The grid search returns the best hyperparameters, the best training R² score (using cross-validation), and the best model (which is saved to disk).

4. **Test Data Setup:**  
   - The test set is defined as the month immediately following the current training period.
   - The test data is extracted for that month and is also preprocessed using the same pipeline.
   - The target values for the test set (y_test) are aligned with the preprocessed X_test to ensure consistency.

5. **Model Evaluation:**  
   - The best model from the grid search is used to predict the target on the test set.
   - The R² score (test_r2) is computed for the test predictions, which provides an indication of how well the model generalizes to new data.

6. **Results Collection:**  
   - For each training period (or each "roll" of the backtest), the following are recorded:
     - The end month of the training data (`train_end`).
     - The corresponding test month (`test_month`).
     - The training R² score (`train_r2`) from the grid search.
     - The test R² score (`test_r2`) from predictions on the test set.
     - The best hyperparameters (`best_params`) that were found.
   - These results are appended to a list and eventually compiled into a results DataFrame.

---

### What You Should Expect in the Results

When you run your backtest, you will get a DataFrame where each row corresponds to one “rolling” backtest iteration. Here’s what each column represents:

- **train_end:**  
  The last month used in the training set for that iteration. For instance, if you pool data from January 2010 and use it for training, you’ll see "2010-01" as the training end month.

- **test_month:**  
  The month immediately following the training period, used for testing the model’s predictions (e.g., "2010-02" if the training ended in "2010-01").

- **train_r2:**  
  The R² score obtained during training (via cross-validation in grid search). This shows how well the model fits the training data.

- **test_r2:**  
  The R² score on the test set. This is a key metric that indicates the model’s predictive performance on unseen data. You might notice that test_r2 is lower than train_r2 if there is any overfitting, or similar if the model generalizes well.

- **best_params:**  
  A dictionary of the best hyperparameters found by grid search for that particular training period. This helps you see if the model’s configuration changes over time.

If everything works correctly, you should see a time series of performance metrics. For instance, you might observe trends such as:

- **Consistent performance:**  
  Both train_r2 and test_r2 are similar across different periods, indicating stable model performance.

- **Degradation in test performance:**  
  If test_r2 is significantly lower than train_r2 or shows a declining trend, this might signal that the model is overfitting to the training data or that market conditions are changing.

- **Variability in hyperparameters:**  
  The best_params might vary between iterations, which could indicate sensitivity of your model to different time periods or data regimes.

---

### Final Notes

- **Data Alignment:**  
  Because you’re dropping rows (e.g., through preprocessing steps like drop_na), aligning y with the preprocessed X using `.loc[X.index]` is crucial to avoid errors such as "inconsistent number of samples."

- **Robustness:**  
  You might want to add additional error handling or logging to capture if specific backtest periods are consistently problematic, which could lead to adjustments in your preprocessing or data pooling strategy.

This comprehensive process gives you a robust method for evaluating your model’s performance over time with different training windows, simulating real-world scenarios where you retrain models periodically and test on the next available period.

---

This explanation should help you understand the mechanics of your backtesting code and what kind of output to expect from it.

## Running Tests

This project uses `pytest` for unit tests. After installing the dependencies, run:

```bash
pytest
```

The tests are located in the `tests/` directory and cover utility functions like portfolio evaluation.
