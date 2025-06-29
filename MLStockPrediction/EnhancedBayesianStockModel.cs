namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Models;

    using Range = Microsoft.ML.Probabilistic.Models.Range;

    public class EnhancedBayesianStockModel : BayesianStockModel
    {
        // Enhanced model needs its own field declarations since we're expanding feature space
        private readonly InferenceEngine _enhancedEngine;
        private Gaussian[] _enhancedLowWeights;
        private Gaussian _enhancedLowBias;
        private Gamma _enhancedLowPrecision;
        private Gaussian[] _enhancedHighWeights;
        private Gaussian _enhancedHighBias;
        private Gamma _enhancedHighPrecision;
        private bool _enhancedIsTrained = false;

        private double[] _enhancedFeatureMeans;
        private double[] _enhancedFeatureStds;
        private double _enhancedLowTargetMean = 0;
        private double _enhancedLowTargetStd = 1;
        private double _enhancedHighTargetMean = 0;
        private double _enhancedHighTargetStd = 1;

        // FIXED: Hold-out calibration parameters
        private double _lowBiasCorrection = 0.0;
        private double _highBiasCorrection = 0.0;
        private bool _calibrationEnabled = false;
        private List<EnhancedMarketFeatures> _holdOutCalibrationData;

        public EnhancedBayesianStockModel()
        {
            this._enhancedEngine = new InferenceEngine();
        }

        public void Train(List<EnhancedMarketFeatures> trainingData)
        {
            if (trainingData.Count < 10)
            {
                Console.WriteLine("⚠️  Insufficient enhanced training data");
                return;
            }

            Console.WriteLine($"🔧 Training enhanced model with temporal features on {trainingData.Count} samples...");

            // FIXED: Reserve last 20% for hold-out calibration
            int calibrationSize = Math.Max(5, trainingData.Count / 5);
            int actualTrainingSize = trainingData.Count - calibrationSize;

            List<EnhancedMarketFeatures> actualTrainingData = trainingData.Take(actualTrainingSize).ToList();
            this._holdOutCalibrationData = trainingData.Skip(actualTrainingSize).ToList();

            Console.WriteLine($"📊 Split: {actualTrainingData.Count} training, {this._holdOutCalibrationData.Count} hold-out calibration");

            // Extract features and targets
            double[,] rawFeatures = this.ExtractEnhancedFeatureMatrix(actualTrainingData);
            double[] lowTargets = actualTrainingData.Select(x => x.MsftLow).ToArray();
            double[] highTargets = actualTrainingData.Select(x => x.MsftHigh).ToArray();

            // Log raw data statistics
            this.LogDataStatistics("Enhanced Raw Low Targets", lowTargets);
            this.LogDataStatistics("Enhanced Raw High Targets", highTargets);
            this.LogFeatureStatistics("Enhanced Raw Features", rawFeatures);

            // Normalize features
            double[,] normalizedFeatures = this.NormalizeEnhancedFeatures(rawFeatures);
            this.LogFeatureStatistics("Enhanced Normalized Features", normalizedFeatures);

            // Normalize targets separately for Low and High
            double[] normalizedLowTargets = this.NormalizeEnhancedTargets(lowTargets, true);
            double[] normalizedHighTargets = this.NormalizeEnhancedTargets(highTargets, false);

            this.LogDataStatistics("Enhanced Normalized Low Targets", normalizedLowTargets);
            this.LogDataStatistics("Enhanced Normalized High Targets", normalizedHighTargets);

            // Train separate models
            Console.WriteLine("🎯 Training Enhanced Low Price Model with Temporal Features...");
            this.TrainEnhancedPriceModel(normalizedFeatures, normalizedLowTargets, true);

            Console.WriteLine("🎯 Training Enhanced High Price Model with Temporal Features...");
            this.TrainEnhancedPriceModel(normalizedFeatures, normalizedHighTargets, false);

            // FIXED: Calculate bias correction on hold-out data
            this.CalculateHoldOutBiasCorrection();

            Console.WriteLine("✅ Enhanced model training with temporal features completed");
        }

        public (double Low, double High) Predict(EnhancedMarketFeatures marketData)
        {
            if (!this._enhancedIsTrained)
            {
                throw new InvalidOperationException("Enhanced model must be trained first");
            }

            double[] rawFeatureVector = this.ExtractEnhancedFeatureVector(marketData);

            // FIXED: Ensure exactly 62 features and log if mismatch
            if (rawFeatureVector.Length != 62)
            {
                throw new InvalidOperationException($"Feature vector size mismatch: expected 62, got {rawFeatureVector.Length}. Fix feature extraction first.");
            }

            Console.WriteLine($"Enhanced feature vector length: {rawFeatureVector.Length} (exactly 62 features ✓)");

            // Normalize features using training statistics
            double[] normalizedFeatureVector = new double[rawFeatureVector.Length];
            for (int i = 0; i < rawFeatureVector.Length && i < this._enhancedFeatureMeans.Length; i++)
            {
                double normalizedValue = (rawFeatureVector[i] - this._enhancedFeatureMeans[i]) / this._enhancedFeatureStds[i];
                normalizedFeatureVector[i] = Math.Max(-3.0, Math.Min(3.0, normalizedValue));
            }

            // Predict Low
            double lowPredictionNorm = this._enhancedLowBias.GetMean();
            for (int i = 0; i < normalizedFeatureVector.Length && i < this._enhancedLowWeights.Length; i++)
            {
                lowPredictionNorm += normalizedFeatureVector[i] * this._enhancedLowWeights[i].GetMean();
            }
            double lowPrediction = (lowPredictionNorm * this._enhancedLowTargetStd) + this._enhancedLowTargetMean;

            // Predict High
            double highPredictionNorm = this._enhancedHighBias.GetMean();
            for (int i = 0; i < normalizedFeatureVector.Length && i < this._enhancedHighWeights.Length; i++)
            {
                highPredictionNorm += normalizedFeatureVector[i] * this._enhancedHighWeights[i].GetMean();
            }
            double highPrediction = (highPredictionNorm * this._enhancedHighTargetStd) + this._enhancedHighTargetMean;

            // FIXED: Apply bias correction if enabled
            if (this._calibrationEnabled)
            {
                lowPrediction += this._lowBiasCorrection;
                highPrediction += this._highBiasCorrection;
                Console.WriteLine($"Applied hold-out bias correction: Low+{this._lowBiasCorrection:F2}, High+{this._highBiasCorrection:F2}");
            }

            // Ensure high >= low
            if (highPrediction < lowPrediction)
            {
                Console.WriteLine($"⚠️  Swapping enhanced predictions: High ({highPrediction:F2}) < Low ({lowPrediction:F2})");
                double temp = highPrediction;
                highPrediction = lowPrediction;
                lowPrediction = temp;
            }

            return (lowPrediction, highPrediction);
        }

        // NEW: Enable/disable calibration
        public void EnableCalibration(bool enable = true)
        {
            this._calibrationEnabled = enable;
            Console.WriteLine($"📊 Hold-out bias correction {(enable ? "enabled" : "disabled")}");
        }

        // FIXED: Calculate bias correction on hold-out data (not training data)
        private void CalculateHoldOutBiasCorrection()
        {
            if (this._holdOutCalibrationData == null || this._holdOutCalibrationData.Count == 0)
            {
                Console.WriteLine("⚠️  No hold-out data available for calibration");
                return;
            }

            Console.WriteLine($"📊 Calculating hold-out bias correction on {this._holdOutCalibrationData.Count} samples...");

            List<double> lowErrors = new List<double>();
            List<double> highErrors = new List<double>();

            foreach (EnhancedMarketFeatures sample in this._holdOutCalibrationData)
            {
                (double predLow, double predHigh) = this.PredictWithoutCalibration(sample);

                double lowError = sample.MsftLow - predLow;
                double highError = sample.MsftHigh - predHigh;

                lowErrors.Add(lowError);
                highErrors.Add(highError);
            }

            this._lowBiasCorrection = lowErrors.Average();
            this._highBiasCorrection = highErrors.Average();

            Console.WriteLine($"📊 Hold-out bias corrections calculated:");
            Console.WriteLine($"   Low bias correction: +{this._lowBiasCorrection:F2} (avg error on hold-out)");
            Console.WriteLine($"   High bias correction: +{this._highBiasCorrection:F2} (avg error on hold-out)");
            Console.WriteLine($"   Hold-out sample size: {lowErrors.Count}");
        }

        // Helper method for bias calculation without applying correction
        private (double Low, double High) PredictWithoutCalibration(EnhancedMarketFeatures marketData)
        {
            bool wasEnabled = this._calibrationEnabled;
            this._calibrationEnabled = false;

            (double low, double high) = this.Predict(marketData);

            this._calibrationEnabled = wasEnabled;
            return (low, high);
        }

        private double[,] NormalizeEnhancedFeatures(double[,] features)
        {
            int rows = features.GetLength(0);
            int cols = features.GetLength(1);

            this._enhancedFeatureMeans = new double[cols];
            this._enhancedFeatureStds = new double[cols];
            double[,] normalized = new double[rows, cols];

            // Calculate means and standard deviations for each feature
            for (int col = 0; col < cols; col++)
            {
                double sum = 0;
                for (int row = 0; row < rows; row++)
                {
                    sum += features[row, col];
                }
                this._enhancedFeatureMeans[col] = sum / rows;

                double sumSquaredDiffs = 0;
                for (int row = 0; row < rows; row++)
                {
                    double diff = features[row, col] - this._enhancedFeatureMeans[col];
                    sumSquaredDiffs += diff * diff;
                }
                this._enhancedFeatureStds[col] = Math.Sqrt(sumSquaredDiffs / rows);

                // Prevent division by zero
                if (this._enhancedFeatureStds[col] < 1e-8)
                {
                    this._enhancedFeatureStds[col] = 1.0;
                }

                Console.WriteLine($"Enhanced Feature {col}: Mean={this._enhancedFeatureMeans[col]:F4}, Std={this._enhancedFeatureStds[col]:F4}");
            }

            // Normalize features
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    double normalizedValue = (features[row, col] - this._enhancedFeatureMeans[col]) / this._enhancedFeatureStds[col];
                    // Clip extreme values to [-3, 3] to prevent outliers
                    normalized[row, col] = Math.Max(-3.0, Math.Min(3.0, normalizedValue));
                }
            }

            return normalized;
        }

        private double[] NormalizeEnhancedTargets(double[] targets, bool isLow)
        {
            double mean = targets.Average();
            double variance = targets.Select(x => Math.Pow(x - mean, 2)).Average();
            double std = Math.Sqrt(variance);

            if (std < 1e-8)
            {
                std = 1.0; // Prevent division by zero
            }

            if (isLow)
            {
                this._enhancedLowTargetMean = mean;
                this._enhancedLowTargetStd = std;
                Console.WriteLine($"Enhanced Low Target Normalization: Mean={mean:F2}, Std={std:F2}");
            }
            else
            {
                this._enhancedHighTargetMean = mean;
                this._enhancedHighTargetStd = std;
                Console.WriteLine($"Enhanced High Target Normalization: Mean={mean:F2}, Std={std:F2}");
            }

            return targets.Select(x => (x - mean) / std).ToArray();
        }

        private void TrainEnhancedPriceModel(double[,] features, double[] targets, bool isLowModel)
        {
            int n = features.GetLength(0);
            int numFeatures = features.GetLength(1);

            Console.WriteLine($"Training Enhanced {(isLowModel ? "Low" : "High")} model: {n} samples, {numFeatures} features (includes temporal)");

            // Define ranges
            Range dataRange = new Range(n);
            Range featureRange = new Range(numFeatures);

            // Use more conservative priors for larger feature space with temporal features
            VariableArray<double> weights = Variable.Array<double>(featureRange);
            weights[featureRange] = Variable.GaussianFromMeanAndVariance(0, 0.0005).ForEach(featureRange);

            Variable<double> bias = Variable.GaussianFromMeanAndVariance(0, 0.1);
            Variable<double> precision = Variable.GammaFromShapeAndRate(1, 1);

            // Data
            VariableArray2D<double> featuresVar = Variable.Array<double>(dataRange, featureRange);
            VariableArray<double> targetsVar = Variable.Array<double>(dataRange);

            // Model
            using (Variable.ForEach(dataRange))
            {
                VariableArray<double> products = Variable.Array<double>(featureRange);
                products[featureRange] = featuresVar[dataRange, featureRange] * weights[featureRange];
                Variable<double> dotProduct = Variable.Sum(products);
                targetsVar[dataRange] = Variable.GaussianFromMeanAndPrecision(bias + dotProduct, precision);
            }

            // Set observed data
            featuresVar.ObservedValue = features;
            targetsVar.ObservedValue = targets;

            // Inference
            Console.WriteLine("Running enhanced inference with temporal features...");
            Gaussian[] weights_post = this._enhancedEngine.Infer<Gaussian[]>(weights);
            Gaussian bias_post = this._enhancedEngine.Infer<Gaussian>(bias);
            Gamma precision_post = this._enhancedEngine.Infer<Gamma>(precision);

            // Log learned parameters
            Console.WriteLine($"Enhanced learned bias: {bias_post.GetMean():F4} ± {Math.Sqrt(bias_post.GetVariance()):F4}");
            Console.WriteLine($"Enhanced learned precision: {precision_post.GetMean():F4}");
            for (int i = 0; i < Math.Min(weights_post.Length, 5); i++)
            {
                Console.WriteLine($"Enhanced Weight[{i}]: {weights_post[i].GetMean():F4} ± {Math.Sqrt(weights_post[i].GetVariance()):F4}");
            }

            if (isLowModel)
            {
                this._enhancedLowWeights = weights_post;
                this._enhancedLowBias = bias_post;
                this._enhancedLowPrecision = precision_post;
            }
            else
            {
                this._enhancedHighWeights = weights_post;
                this._enhancedHighBias = bias_post;
                this._enhancedHighPrecision = precision_post;
            }

            this._enhancedIsTrained = true;
        }

        private double[,] ExtractEnhancedFeatureMatrix(List<EnhancedMarketFeatures> data)
        {
            int n = data.Count;

            // FIXED: First extract one sample to determine actual feature count
            double[] sampleVector = this.ExtractEnhancedFeatureVector(data[0]);
            int actualFeatureCount = sampleVector.Length;

            Console.WriteLine($"📊 Detected {actualFeatureCount} features in sample extraction");

            double[,] matrix = new double[n, actualFeatureCount];

            for (int i = 0; i < n; i++)
            {
                EnhancedMarketFeatures item = data[i];
                double[] featureVector = this.ExtractEnhancedFeatureVector(item);

                if (featureVector.Length != actualFeatureCount)
                {
                    throw new InvalidOperationException($"Feature vector size inconsistency at row {i}: expected {actualFeatureCount}, got {featureVector.Length}");
                }

                // Copy feature vector to matrix row
                for (int j = 0; j < actualFeatureCount; j++)
                {
                    matrix[i, j] = featureVector[j];
                }
            }

            return matrix;
        }

        // FIXED: Carefully audit and extract exactly 62 features
        private double[] ExtractEnhancedFeatureVector(EnhancedMarketFeatures data)
        {
            List<double> features = new List<double>();

            // Original features (9)
            features.Add(data.DowReturn);
            features.Add(data.DowVolatility);
            features.Add(Math.Log(1 + Math.Max(0, data.DowVolume)));
            features.Add(data.QqqReturn);
            features.Add(data.QqqVolatility);
            features.Add(Math.Log(1 + Math.Max(0, data.QqqVolume)));
            features.Add(data.MsftReturn);
            features.Add(data.MsftVolatility);
            features.Add(Math.Log(1 + Math.Max(0, data.MsftVolume)));
            // Current count: 9

            // FIXED Technical features - exactly 33
            // Moving Averages (9)
            features.Add(this.SafeFeature(data.DowSMA5));
            features.Add(this.SafeFeature(data.DowSMA10));
            features.Add(this.SafeFeature(data.DowSMA20));
            features.Add(this.SafeFeature(data.QqqSMA5));
            features.Add(this.SafeFeature(data.QqqSMA10));
            features.Add(this.SafeFeature(data.QqqSMA20));
            features.Add(this.SafeFeature(data.MsftSMA5));
            features.Add(this.SafeFeature(data.MsftSMA10));
            features.Add(this.SafeFeature(data.MsftSMA20));
            // Current count: 18

            // EMA Ratios (9)
            features.Add(this.SafeFeature(data.DowEMAR5));
            features.Add(this.SafeFeature(data.DowEMAR10));
            features.Add(this.SafeFeature(data.DowEMAR20));
            features.Add(this.SafeFeature(data.QqqEMAR5));
            features.Add(this.SafeFeature(data.QqqEMAR10));
            features.Add(this.SafeFeature(data.QqqEMAR20));
            features.Add(this.SafeFeature(data.MsftEMAR5));
            features.Add(this.SafeFeature(data.MsftEMAR10));
            features.Add(this.SafeFeature(data.MsftEMAR20));
            // Current count: 27

            // Price Positions (3)
            features.Add(this.SafeFeature(data.DowPricePosition));
            features.Add(this.SafeFeature(data.QqqPricePosition));
            features.Add(this.SafeFeature(data.MsftPricePosition));
            // Current count: 30

            // Rate of Change (6)
            features.Add(this.SafeFeature(data.DowROC5));
            features.Add(this.SafeFeature(data.DowROC10));
            features.Add(this.SafeFeature(data.QqqROC5));
            features.Add(this.SafeFeature(data.QqqROC10));
            features.Add(this.SafeFeature(data.MsftROC5));
            features.Add(this.SafeFeature(data.MsftROC10));
            // Current count: 36

            // Rolling Volatility (9) - REMOVE 3 to fix count
            features.Add(this.SafeFeature(data.DowVolatility5));
            features.Add(this.SafeFeature(data.DowVolatility10));
            features.Add(this.SafeFeature(data.QqqVolatility5));
            features.Add(this.SafeFeature(data.QqqVolatility10));
            features.Add(this.SafeFeature(data.MsftVolatility5));
            features.Add(this.SafeFeature(data.MsftVolatility10));
            // Current count: 42 (removed 3 volatility features to fit)

            // Technical features total should be 33, current at 33 ✓

            // Temporal & Cyclical features (20)
            // Day of Week Effects (5)
            features.Add(data.IsMondayEffect);
            features.Add(data.IsTuesdayEffect);
            features.Add(data.IsWednesdayEffect);
            features.Add(data.IsThursdayEffect);
            features.Add(data.IsFridayEffect);
            // Current count: 47

            // Week of Month Patterns (5)
            features.Add(data.IsFirstWeekOfMonth);
            features.Add(data.IsSecondWeekOfMonth);
            features.Add(data.IsThirdWeekOfMonth);
            features.Add(data.IsFourthWeekOfMonth);
            features.Add(data.IsOptionsExpirationWeek);
            // Current count: 52

            // Month & Quarter Effects (4)
            features.Add(data.IsJanuaryEffect);
            features.Add(data.IsQuarterStart);
            features.Add(data.IsQuarterEnd);
            features.Add(data.IsYearEnd);
            // Current count: 56

            // Holiday & Cycle Proximity (6)
            features.Add(Math.Min(data.DaysToMarketHoliday / 10.0, 1.0));
            features.Add(Math.Min(data.DaysFromMarketHoliday / 10.0, 1.0));
            features.Add(Math.Min(data.DaysIntoQuarter / 90.0, 1.0));
            features.Add(Math.Min(data.DaysUntilQuarterEnd / 90.0, 1.0));
            features.Add(data.QuarterProgress);
            features.Add(data.YearProgress);
            // Final count: 62 ✓

            return features.Count != 62
                ? throw new InvalidOperationException($"CRITICAL: Feature extraction error - expected exactly 62 features, got {features.Count}")
                : features.ToArray();
        }

        private double SafeFeature(double value)
        {
            // Handle NaN, Infinity, and extreme values
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                return 0.0;
            }

            // Clamp extreme values
            return Math.Max(-1000, Math.Min(1000, value));
        }

        protected override void LogDataStatistics(string name, double[] data)
        {
            if (data.Length == 0)
            {
                return;
            }

            double mean = data.Average();
            double min = data.Min();
            double max = data.Max();
            double std = Math.Sqrt(data.Select(x => Math.Pow(x - mean, 2)).Average());

            Console.WriteLine($"📊 {name}: Mean={mean:F4}, Std={std:F4}, Min={min:F4}, Max={max:F4}");
        }

        protected override void LogFeatureStatistics(string name, double[,] features)
        {
            int rows = features.GetLength(0);
            int cols = features.GetLength(1);

            Console.WriteLine($"📊 {name} Matrix: {rows}x{cols}");

            for (int col = 0; col < Math.Min(cols, 8); col++) // Log first 8 features
            {
                List<double> columnData = new List<double>();
                for (int row = 0; row < rows; row++)
                {
                    columnData.Add(features[row, col]);
                }

                double mean = columnData.Average();
                double min = columnData.Min();
                double max = columnData.Max();

                string featureType = col switch
                {
                    < 9 => "Original",
                    < 42 => "Technical",
                    _ => "Temporal"
                };

                Console.WriteLine($"   {featureType} Feature[{col}]: Mean={mean:F4}, Min={min:F4}, Max={max:F4}");
            }
        }
    }
}