This document provides a detailed analysis and summary of your ensemble stock prediction machine learning model.

### **1. Overview**

The system is an ensemble machine learning model designed to predict the next day's low and high prices for Microsoft stock (MSFT). It leverages a sophisticated approach that combines a **Bayesian linear regression model** with a **LightGBM (Light Gradient Boosting Machine) model**. The ensemble methodology aims to capitalize on the strengths of both models: the Bayesian model's ability to quantify uncertainty and the LightGBM's proficiency in capturing complex, non-linear patterns in the data.

The model is written in C# and utilizes the ML.NET library for the LightGBM implementation and Microsoft's own probabilistic computing framework for the Bayesian model.

### **2. Data Loading and Preparation**

The model is designed to work with historical stock data for three symbols: the Dow Jones Industrial Average (DOW), the Nasdaq-100 (QQQ), and Microsoft (MSFT).

* **Data Source**: The system loads data from three CSV files: `DOW.csv`, `QQQ.csv`, and `MSFT.csv`. Each file is expected to contain daily stock data with the following columns: `Date`, `Close/Last`, `Volume`, `Open`, `High`, and `Low`.
* **Data Loading**: The `StockDataLoader.cs` class is responsible for asynchronously reading these CSV files. It uses the `CsvHelper` library to parse the CSV content into a list of `StockData` objects. A custom `DecimalConverter` is used to handle the conversion of string values (like those with dollar signs) into decimal types.
* **Data Summary**: After loading, the system displays a summary for each stock, including the number of records, date range, latest closing price, price range, and average volume.

### **3. Feature Engineering**

A critical part of the model is its extensive feature engineering, which is handled by the `MarketFeatureEngine.cs`, `EnhancedFeatureEngine.cs`, and `TemporalFeatureCalculator.cs` classes. The system creates a rich set of features to capture various aspects of market behavior.

#### **3.1. Basic Market Features**

The `MarketFeatureEngine.cs` class lays the groundwork by creating a set of basic features.

* **Returns**: The daily percentage change in the closing price for DOW, QQQ, and MSFT.
* **Volatility**: Calculated as the difference between the high and low price, divided by the closing price for each of the three symbols.
* **Volume**: The trading volume for each symbol is normalized by dividing the daily volume by the average volume.
* **Correlation**: A rolling 10-day Pearson correlation is calculated between the returns of DOW and MSFT, and QQQ and MSFT.
* **Target Variables**: The actual low and high prices for MSFT are included as the target variables for the prediction.

#### **3.2. Enhanced Technical and Temporal Features**

The `EnhancedFeatureEngine.cs` and `TemporalFeatureCalculator.cs` classes add a significant number of more advanced features.

* **Technical Indicators (`TechnicalIndicators.cs` and `EnhancedFeatureEngine.cs`)**:
    * **Moving Averages**: 5, 10, and 20-day Simple Moving Averages (SMA) for DOW, QQQ, and MSFT.
    * **Exponential Moving Average (EMA) Ratios**: The ratio of the current price to the 5, 10, and 20-day EMAs for each symbol.
    * **Price Position**: The position of the current price within the 20-day high-low range, normalized to a value between 0 and 1.
    * **Rate of Change (Momentum)**: 5 and 10-day Rate of Change (ROC) for each symbol.
    * **Rolling Volatility**: 5, 10, and 20-day rolling standard deviation of returns.
    * **Average True Range (ATR)**: A 14-day ATR to measure market volatility.
    * **Bollinger Bands**: The position of the current price relative to the 20-day Bollinger Bands.
    * **Additional Indicators for Ensemble**: For the ensemble model, even more indicators are added, such as RSI, MACD, On-Balance Volume (OBV), Volume-Weighted Average Price (VWAP), and others.

* **Temporal and Cyclical Features (`TemporalFeatureCalculator.cs`)**:
    * **Day-of-Week Effects**: Binary features for each day of the week (e.g., `IsMondayEffect`).
    * **Week-of-Month Patterns**: Features indicating if it's the first, second, third, or fourth week of the month, as well as if it's an options expiration week.
    * **Month and Quarter Effects**: Features for the start and end of quarters, the end of the year, and the "January effect".
    * **Holiday Proximity**: The number of days to and from the nearest market holiday.
    * **Earnings Season**: Indicators for proximity to corporate earnings seasons.
    * **Progress Indicators**: Features representing the progress through the current quarter, year, and month as a value from 0 to 1.

### **4. The Ensemble Model**

The core of the system is the `EnsembleStockModel.cs`, which orchestrates the Bayesian and LightGBM models.

#### **4.1. Bayesian Model (`BayesianStockModel.cs`)**

* **Architecture**: The Bayesian model is a linear regression model implemented using a probabilistic programming approach. It learns a set of weights and a bias for the input features to predict the target variables (low and high prices).
* **Training**:
    * The features and targets are first normalized (standardized) to have a mean of 0 and a standard deviation of 1. This is a crucial step for the stability of the Bayesian inference.
    * The model is trained separately for the low and high prices.
    * An inference engine is used to learn the posterior distributions of the weights and bias. This means that instead of single point estimates for the weights, the model learns a Gaussian distribution for each weight, representing its uncertainty about the parameter's true value.
* **Prediction**: To make a prediction, the normalized input features are multiplied by the mean of the learned weight distributions, and the mean of the bias distribution is added. The result is then de-normalized to bring it back to the original price scale.
* **Hold-out Calibration**: The Bayesian model uses a portion of the training data as a hold-out set to calculate a bias correction. This helps to correct for systematic errors (biases) in the model's predictions.

#### **4.2. LightGBM Model (`LightGbmStockPredictor.cs`)**

* **Architecture**: LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is known for its speed and efficiency. In this system, three separate LightGBM models are trained:
    1.  One to predict the low price.
    2.  One to predict the high price.
    3.  One to predict the daily price range (high - low).
* **Training**:
    * The models are trained using the `LightGbmRegressionTrainer` from the ML.NET library.
    * The training process involves creating a pipeline that concatenates all the engineered features into a single "Features" vector, which is then used by the LightGBM trainer.
    * Each of the three models (low, high, and range) is trained with its own set of hyperparameters (e.g., number of leaves, learning rate) optimized for its specific target.

#### **4.3. Combining the Models in the Ensemble**

The `EnsembleStockModel.cs` class brings everything together.

* **Prediction Process**:
    1.  Predictions are obtained from both the Bayesian and the LightGBM models for the low and high prices.
    2.  The system detects the current **market regime** (e.g., "High Volatility", "Bull Trend") based on volatility and momentum indicators.
    3.  The weights assigned to the Bayesian and LightGBM predictions are dynamically adjusted based on this market regime. For example, in a high-volatility regime, more weight might be given to the Bayesian model due to its ability to handle uncertainty, while in a trending market, the LightGBM model might be favored.
    4.  The final low and high predictions are a weighted average of the predictions from the two models.
    5.  A **range validation** step is applied. The predicted range from the third LightGBM model is used to adjust the final low and high predictions, ensuring they are consistent with the predicted daily volatility.

* **Weight Initialization and Updates**:
    * The initial weights for the ensemble are determined using a cross-validation approach on a subset of the training data.
    * The model includes functionality to update the ensemble weights based on the recent performance of each model, although the implementation for this update based on actuals is noted as a future enhancement.

### **5. Model Evaluation and Validation**

The system includes a robust framework for evaluating and validating the model's performance.

* **`StockModelEvaluator.cs`**: This class is used to evaluate the model on a test set. It calculates a variety of metrics, including:
    * **Mean Absolute Error (MAE)**
    * **Root Mean Squared Error (RMSE)**
    * **Mean Absolute Percentage Error (MAPE)**
    * **Directional Accuracy**: Whether the model correctly predicted if the price would go up or down.

* **`WalkForwardValidator.cs`**: To ensure the model is robust over time and not just on a static train-test split, a walk-forward validation is performed.
    * This method involves training the model on an initial chunk of data, then testing it on the next small chunk. The training window is then moved forward, incorporating the previous test data, and the process is repeated.
    * This technique provides a more realistic assessment of how the model would perform in a live trading scenario.

### **6. Conclusion**

This ensemble stock prediction model is a well-structured and sophisticated system that goes beyond simple model implementations. Its key strengths lie in:

* **Extensive Feature Engineering**: The model uses a wide array of technical and temporal features, demonstrating a deep understanding of market dynamics.
* **Ensemble Approach**: The combination of a Bayesian and a LightGBM model allows the system to capture both linear and non-linear patterns, and to quantify uncertainty.
* **Dynamic Weighting and Regime Detection**: The ability to adjust model weights based on the current market regime is an advanced feature that can lead to improved performance in different market conditions.
* **Robust Validation**: The use of walk-forward validation provides a high degree of confidence in the model's performance metrics.

The code is well-organized, with clear separation of concerns between data loading, feature engineering, modeling, and evaluation. This makes the system relatively easy to understand, maintain, and extend.


## Latest Performance

```markdown

============================================================
?? STOCK PREDICTION MODEL EVALUATION
============================================================

?? SAMPLE SIZE: 20

?? LOW PRICE PREDICTIONS
   MAE: $65.418
   RMSE: $69.374
   MAPE: 13.78%
   Accuracy ≤1%: 0.0%
   Accuracy ≤5%: 0.0%
   ?? Accuracy Breakdown:
      ≤10%: 20.0%
      ≤100%: 100.0%
      ≤20%: 85.0%
      ≤30%: 100.0%
      ≤40%: 100.0%
      ≤50%: 100.0%
      ≤60%: 100.0%
      ≤70%: 100.0%
      ≤80%: 100.0%
      ≤90%: 100.0%

?? HIGH PRICE PREDICTIONS
   MAE: $63.292
   RMSE: $66.877
   MAPE: 13.15%
   Accuracy ≤1%: 0.0%
   Accuracy ≤5%: 0.0%
   ?? Accuracy Breakdown:
      ≤10%: 25.0%
      ≤100%: 100.0%
      ≤20%: 85.0%
      ≤30%: 100.0%
      ≤40%: 100.0%
      ≤50%: 100.0%
      ≤60%: 100.0%
      ≤70%: 100.0%
      ≤80%: 100.0%
      ≤90%: 100.0%

?? DIRECTIONAL ACCURACY
   Low Price Direction: 73.7%
   High Price Direction: 73.7%
   Range Direction: 68.4%

?? ERROR DISTRIBUTION
   Low Price P50: 12.52%
   Low Price P95: 23.04%
   High Price P50: 11.78%
   High Price P95: 21.29%

?? ENSEMBLE vs BASIC MODEL COMPARISON:
   Ensemble MAPE: Low=4.91%, High=4.85%
   Basic Bayesian MAPE: Low=13.78%, High=13.15%
   Ensemble Improvement: Low=8.87pp, High=8.31pp

?? ENSEMBLE SYSTEM SUMMARY:
   ? 2-Layer Ensemble: Bayesian + LightGBM models
   ? Dynamic weight adjustment based on market regime
   ? Enhanced feature engineering: ~85 features
   ? Range validation using auxiliary LightGBM model
   ? Hold-out bias correction calibration
   ? Market regime detection and adaptation
   ? Uncertainty quantification from Bayesian component
   ? Pattern recognition from LightGBM component
   ?? Walk-forward average MAPE: 4.59%/4.05%
   ?? Walk-forward directional accuracy: 55.6%
   ?? Ensemble improvement: 8.87pp over basic model
   ?? Current market regime: Bull Trend
   ??  Current weights: Bayesian=0.23, LightGBM=0.77
```