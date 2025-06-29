namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Models;

    using MLStockPrediction.Models;

    using Range = Microsoft.ML.Probabilistic.Models.Range;

    public class BayesianStockModel
    {
        private readonly InferenceEngine _engine;
        private Gaussian[] _lowWeights;
        private Gaussian _lowBias;
        private Gamma _lowPrecision;
        private Gaussian[] _highWeights;
        private Gaussian _highBias;
        private Gamma _highPrecision;
        private bool _isTrained = false;

        // Normalization parameters
        private double[] _featureMeans;
        private double[] _featureStds;
        private double _lowTargetMean = 0;
        private double _lowTargetStd = 1;
        private double _highTargetMean = 0;
        private double _highTargetStd = 1;

        // Hold-out calibration parameters
        private double _lowBiasCorrection = 0.0;
        private double _highBiasCorrection = 0.0;
        private bool _calibrationEnabled = false;
        private List<EnhancedMarketFeatures> _holdOutCalibrationData;

        public BayesianStockModel()
        {
            this._engine = new InferenceEngine();
        }

        public void Train(List<MarketFeatures> trainingData)
        {
            if (trainingData.Count < 2)
            {
                Console.WriteLine("⚠️  Insufficient training data");
                return;
            }

            Console.WriteLine($"🔧 Training model with {trainingData.Count} samples...");

            // Extract features and targets
            double[,] rawFeatures = this.ExtractFeatureMatrix(trainingData);
            double[] lowTargets = trainingData.Select(x => x.MsftLow).ToArray();
            double[] highTargets = trainingData.Select(x => x.MsftHigh).ToArray();

            this.LogDataStatistics("Raw Low Targets", lowTargets);
            this.LogDataStatistics("Raw High Targets", highTargets);
            this.LogFeatureStatistics("Raw Features", rawFeatures);

            // Normalize features
            double[,] normalizedFeatures = this.NormalizeFeatures(rawFeatures);
            this.LogFeatureStatistics("Normalized Features", normalizedFeatures);

            // Normalize targets separately for Low and High
            double[] normalizedLowTargets = this.NormalizeTargets(lowTargets, true);
            double[] normalizedHighTargets = this.NormalizeTargets(highTargets, false);

            this.LogDataStatistics("Normalized Low Targets", normalizedLowTargets);
            this.LogDataStatistics("Normalized High Targets", normalizedHighTargets);

            // Train separate models
            Console.WriteLine("🎯 Training Low Price Model...");
            this.TrainPriceModel(normalizedFeatures, normalizedLowTargets, true);

            Console.WriteLine("🎯 Training High Price Model...");
            this.TrainPriceModel(normalizedFeatures, normalizedHighTargets, false);

            Console.WriteLine("✅ Model training completed");
        }

        public void Train(List<EnhancedMarketFeatures> trainingData)
        {
            if (trainingData.Count < 10)
            {
                Console.WriteLine("⚠️  Insufficient enhanced training data");
                return;
            }

            Console.WriteLine($"🔧 Training enhanced model with temporal features on {trainingData.Count} samples...");

            // Reserve last 20% for hold-out calibration
            int calibrationSize = Math.Max(5, trainingData.Count / 5);
            int actualTrainingSize = trainingData.Count - calibrationSize;

            List<EnhancedMarketFeatures> actualTrainingData = trainingData.Take(actualTrainingSize).ToList();
            this._holdOutCalibrationData = trainingData.Skip(actualTrainingSize).ToList();

            Console.WriteLine($"📊 Split: {actualTrainingData.Count} training, {this._holdOutCalibrationData.Count} hold-out calibration");

            // Extract features and targets
            double[,] rawFeatures = this.ExtractEnhancedFeatureMatrix(actualTrainingData);
            double[] lowTargets = actualTrainingData.Select(x => x.MsftLow).ToArray();
            double[] highTargets = actualTrainingData.Select(x => x.MsftHigh).ToArray();

            this.LogDataStatistics("Enhanced Raw Low Targets", lowTargets);
            this.LogDataStatistics("Enhanced Raw High Targets", highTargets);
            this.LogFeatureStatistics("Enhanced Raw Features", rawFeatures);

            // Normalize features
            double[,] normalizedFeatures = this.NormalizeFeatures(rawFeatures);
            this.LogFeatureStatistics("Enhanced Normalized Features", normalizedFeatures);

            // Normalize targets separately for Low and High
            double[] normalizedLowTargets = this.NormalizeTargets(lowTargets, true);
            double[] normalizedHighTargets = this.NormalizeTargets(highTargets, false);

            this.LogDataStatistics("Enhanced Normalized Low Targets", normalizedLowTargets);
            this.LogDataStatistics("Enhanced Normalized High Targets", normalizedHighTargets);

            // Train separate models
            Console.WriteLine("🎯 Training Enhanced Low Price Model with Temporal Features...");
            this.TrainEnhancedPriceModel(normalizedFeatures, normalizedLowTargets, true);

            Console.WriteLine("🎯 Training Enhanced High Price Model with Temporal Features...");
            this.TrainEnhancedPriceModel(normalizedFeatures, normalizedHighTargets, false);

            // Calculate bias correction on hold-out data
            this.CalculateHoldOutBiasCorrection();

            Console.WriteLine("✅ Enhanced model training with temporal features completed");
        }

        public (double Low, double High) Predict(MarketFeatures marketData)
        {
            if (!this._isTrained)
            {
                throw new InvalidOperationException("Model must be trained first");
            }

            double[] rawFeatureVector = this.ExtractFeatureVector(marketData);
            Console.WriteLine($"Raw feature vector: [{string.Join(", ", rawFeatureVector.Select(x => $"{x:F4}"))}]");

            // Normalize features using training statistics
            double[] normalizedFeatureVector = new double[rawFeatureVector.Length];
            for (int i = 0; i < rawFeatureVector.Length && i < this._featureMeans.Length; i++)
            {
                double normalizedValue = (rawFeatureVector[i] - this._featureMeans[i]) / this._featureStds[i];
                normalizedFeatureVector[i] = Math.Max(-3.0, Math.Min(3.0, normalizedValue));
            }
            Console.WriteLine($"Normalized feature vector: [{string.Join(", ", normalizedFeatureVector.Select(x => $"{x:F4}"))}]");

            return this.PredictInternal(normalizedFeatureVector);
        }

        public (double Low, double High) Predict(EnhancedMarketFeatures marketData)
        {
            if (!this._isTrained)
            {
                throw new InvalidOperationException("Enhanced model must be trained first");
            }

            double[] rawFeatureVector = this.ExtractEnhancedFeatureVector(marketData);

            // Ensure exactly 62 features
            if (rawFeatureVector.Length != 62)
            {
                throw new InvalidOperationException($"Feature vector size mismatch: expected 62, got {rawFeatureVector.Length}. Fix feature extraction first.");
            }

            Console.WriteLine($"Enhanced feature vector length: {rawFeatureVector.Length} (exactly 62 features ✓)");

            // Normalize features using training statistics
            double[] normalizedFeatureVector = new double[rawFeatureVector.Length];
            for (int i = 0; i < rawFeatureVector.Length && i < this._featureMeans.Length; i++)
            {
                double normalizedValue = (rawFeatureVector[i] - this._featureMeans[i]) / this._featureStds[i];
                normalizedFeatureVector[i] = Math.Max(-3.0, Math.Min(3.0, normalizedValue));
            }

            return this.PredictInternal(normalizedFeatureVector);
        }

        public void EnableCalibration(bool enable = true)
        {
            this._calibrationEnabled = enable;
            Console.WriteLine($"📊 Hold-out bias correction {(enable ? "enabled" : "disabled")}");
        }

        private (double Low, double High) PredictInternal(double[] normalizedFeatureVector)
        {
            // Predict Low
            double lowPredictionNorm = this._lowBias.GetMean();
            for (int i = 0; i < normalizedFeatureVector.Length && i < this._lowWeights.Length; i++)
            {
                lowPredictionNorm += normalizedFeatureVector[i] * this._lowWeights[i].GetMean();
            }
            double lowPrediction = (lowPredictionNorm * this._lowTargetStd) + this._lowTargetMean;

            // Predict High
            double highPredictionNorm = this._highBias.GetMean();
            for (int i = 0; i < normalizedFeatureVector.Length && i < this._highWeights.Length; i++)
            {
                highPredictionNorm += normalizedFeatureVector[i] * this._highWeights[i].GetMean();
            }
            double highPrediction = (highPredictionNorm * this._highTargetStd) + this._highTargetMean;

            // Apply bias correction if enabled
            if (this._calibrationEnabled)
            {
                lowPrediction += this._lowBiasCorrection;
                highPrediction += this._highBiasCorrection;
                Console.WriteLine($"Applied hold-out bias correction: Low+{this._lowBiasCorrection:F2}, High+{this._highBiasCorrection:F2}");
            }

            // Ensure high >= low
            if (highPrediction < lowPrediction)
            {
                Console.WriteLine($"⚠️  Swapping predictions: High ({highPrediction:F2}) < Low ({lowPrediction:F2})");
                (lowPrediction, highPrediction) = (highPrediction, lowPrediction);
            }

            return (lowPrediction, highPrediction);
        }

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

        private (double Low, double High) PredictWithoutCalibration(EnhancedMarketFeatures marketData)
        {
            bool wasEnabled = this._calibrationEnabled;
            this._calibrationEnabled = false;

            (double low, double high) = this.Predict(marketData);

            this._calibrationEnabled = wasEnabled;
            return (low, high);
        }

        private double[,] NormalizeFeatures(double[,] features)
        {
            int rows = features.GetLength(0);
            int cols = features.GetLength(1);

            this._featureMeans = new double[cols];
            this._featureStds = new double[cols];
            double[,] normalized = new double[rows, cols];

            // Calculate means and standard deviations for each feature
            for (int col = 0; col < cols; col++)
            {
                double sum = 0;
                for (int row = 0; row < rows; row++)
                {
                    sum += features[row, col];
                }
                this._featureMeans[col] = sum / rows;

                double sumSquaredDiffs = 0;
                for (int row = 0; row < rows; row++)
                {
                    double diff = features[row, col] - this._featureMeans[col];
                    sumSquaredDiffs += diff * diff;
                }
                this._featureStds[col] = Math.Sqrt(sumSquaredDiffs / rows);

                // Prevent division by zero
                if (this._featureStds[col] < 1e-8)
                {
                    this._featureStds[col] = 1.0;
                }

                Console.WriteLine($"Feature {col}: Mean={this._featureMeans[col]:F4}, Std={this._featureStds[col]:F4}");
            }

            // Normalize features
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    double normalizedValue = (features[row, col] - this._featureMeans[col]) / this._featureStds[col];
                    normalized[row, col] = Math.Max(-3.0, Math.Min(3.0, normalizedValue));
                }
            }

            return normalized;
        }

        private double[] NormalizeTargets(double[] targets, bool isLow)
        {
            double mean = targets.Average();
            double variance = targets.Select(x => Math.Pow(x - mean, 2)).Average();
            double std = Math.Sqrt(variance);

            if (std < 1e-8)
            {
                std = 1.0;
            }

            if (isLow)
            {
                this._lowTargetMean = mean;
                this._lowTargetStd = std;
                Console.WriteLine($"Low Target Normalization: Mean={mean:F2}, Std={std:F2}");
            }
            else
            {
                this._highTargetMean = mean;
                this._highTargetStd = std;
                Console.WriteLine($"High Target Normalization: Mean={mean:F2}, Std={std:F2}");
            }

            return targets.Select(x => (x - mean) / std).ToArray();
        }

        private void TrainPriceModel(double[,] features, double[] targets, bool isLowModel)
        {
            int n = features.GetLength(0);
            int numFeatures = features.GetLength(1);

            Console.WriteLine($"Training {(isLowModel ? "Low" : "High")} model: {n} samples, {numFeatures} features");

            // Define ranges
            Range dataRange = new Range(n);
            Range featureRange = new Range(numFeatures);

            // Use more conservative priors
            VariableArray<double> weights = Variable.Array<double>(featureRange);
            weights[featureRange] = Variable.GaussianFromMeanAndVariance(0, 0.01).ForEach(featureRange);

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
            Console.WriteLine("Running inference...");
            Gaussian[] weights_post = this._engine.Infer<Gaussian[]>(weights);
            Gaussian bias_post = this._engine.Infer<Gaussian>(bias);
            Gamma precision_post = this._engine.Infer<Gamma>(precision);

            // Log learned parameters
            Console.WriteLine($"Learned bias: {bias_post.GetMean():F4} ± {Math.Sqrt(bias_post.GetVariance()):F4}");
            Console.WriteLine($"Learned precision: {precision_post.GetMean():F4}");
            for (int i = 0; i < Math.Min(weights_post.Length, 5); i++)
            {
                Console.WriteLine($"Weight[{i}]: {weights_post[i].GetMean():F4} ± {Math.Sqrt(weights_post[i].GetVariance()):F4}");
            }

            if (isLowModel)
            {
                this._lowWeights = weights_post;
                this._lowBias = bias_post;
                this._lowPrecision = precision_post;
            }
            else
            {
                this._highWeights = weights_post;
                this._highBias = bias_post;
                this._highPrecision = precision_post;
            }

            this._isTrained = true;
        }

        private void TrainEnhancedPriceModel(double[,] features, double[] targets, bool isLowModel)
        {
            int n = features.GetLength(0);
            int numFeatures = features.GetLength(1);

            Console.WriteLine($"Training Enhanced {(isLowModel ? "Low" : "High")} model: {n} samples, {numFeatures} features (includes temporal)");

            // Define ranges
            Range dataRange = new Range(n);
            Range featureRange = new Range(numFeatures);

            // Use more conservative priors for larger feature space
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
            Gaussian[] weights_post = this._engine.Infer<Gaussian[]>(weights);
            Gaussian bias_post = this._engine.Infer<Gaussian>(bias);
            Gamma precision_post = this._engine.Infer<Gamma>(precision);

            // Log learned parameters
            Console.WriteLine($"Enhanced learned bias: {bias_post.GetMean():F4} ± {Math.Sqrt(bias_post.GetVariance()):F4}");
            Console.WriteLine($"Enhanced learned precision: {precision_post.GetMean():F4}");
            for (int i = 0; i < Math.Min(weights_post.Length, 5); i++)
            {
                Console.WriteLine($"Enhanced Weight[{i}]: {weights_post[i].GetMean():F4} ± {Math.Sqrt(weights_post[i].GetVariance()):F4}");
            }

            if (isLowModel)
            {
                this._lowWeights = weights_post;
                this._lowBias = bias_post;
                this._lowPrecision = precision_post;
            }
            else
            {
                this._highWeights = weights_post;
                this._highBias = bias_post;
                this._highPrecision = precision_post;
            }

            this._isTrained = true;
        }

        private double[,] ExtractFeatureMatrix(List<MarketFeatures> data)
        {
            int n = data.Count;
            int numFeatures = 9;
            double[,] matrix = new double[n, numFeatures];

            for (int i = 0; i < n; i++)
            {
                MarketFeatures item = data[i];
                matrix[i, 0] = item.DowReturn;
                matrix[i, 1] = item.DowVolatility;
                matrix[i, 2] = Math.Log(1 + item.DowVolume);
                matrix[i, 3] = item.QqqReturn;
                matrix[i, 4] = item.QqqVolatility;
                matrix[i, 5] = Math.Log(1 + item.QqqVolume);
                matrix[i, 6] = item.MsftReturn;
                matrix[i, 7] = item.MsftVolatility;
                matrix[i, 8] = Math.Log(1 + item.MsftVolume);
            }

            return matrix;
        }

        private double[,] ExtractEnhancedFeatureMatrix(List<EnhancedMarketFeatures> data)
        {
            int n = data.Count;

            // First extract one sample to determine actual feature count
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

        private double[] ExtractFeatureVector(MarketFeatures data)
        {
            return new double[]
            {
                data.DowReturn,
                data.DowVolatility,
                Math.Log(1 + data.DowVolume),
                data.QqqReturn,
                data.QqqVolatility,
                Math.Log(1 + data.QqqVolume),
                data.MsftReturn,
                data.MsftVolatility,
                Math.Log(1 + data.MsftVolume)
            };
        }

        private double[] ExtractEnhancedFeatureVector(EnhancedMarketFeatures data)
        {
            List<double> features = new List<double>
            {
                // Original features (9)
                data.DowReturn,
                data.DowVolatility,
                Math.Log(1 + Math.Max(0, data.DowVolume)),
                data.QqqReturn,
                data.QqqVolatility,
                Math.Log(1 + Math.Max(0, data.QqqVolume)),
                data.MsftReturn,
                data.MsftVolatility,
                Math.Log(1 + Math.Max(0, data.MsftVolume)),

                // Technical features (33)
                // Moving Averages (9)
                SafeFeature(data.DowSMA5),
                SafeFeature(data.DowSMA10),
                SafeFeature(data.DowSMA20),
                SafeFeature(data.QqqSMA5),
                SafeFeature(data.QqqSMA10),
                SafeFeature(data.QqqSMA20),
                SafeFeature(data.MsftSMA5),
                SafeFeature(data.MsftSMA10),
                SafeFeature(data.MsftSMA20),

                // EMA Ratios (9)
                SafeFeature(data.DowEMAR5),
                SafeFeature(data.DowEMAR10),
                SafeFeature(data.DowEMAR20),
                SafeFeature(data.QqqEMAR5),
                SafeFeature(data.QqqEMAR10),
                SafeFeature(data.QqqEMAR20),
                SafeFeature(data.MsftEMAR5),
                SafeFeature(data.MsftEMAR10),
                SafeFeature(data.MsftEMAR20),

                // Price Positions (3)
                SafeFeature(data.DowPricePosition),
                SafeFeature(data.QqqPricePosition),
                SafeFeature(data.MsftPricePosition),

                // Rate of Change (6)
                SafeFeature(data.DowROC5),
                SafeFeature(data.DowROC10),
                SafeFeature(data.QqqROC5),
                SafeFeature(data.QqqROC10),
                SafeFeature(data.MsftROC5),
                SafeFeature(data.MsftROC10),

                // Rolling Volatility (6 - reduced from 9 to fit)
                SafeFeature(data.DowVolatility5),
                SafeFeature(data.DowVolatility10),
                SafeFeature(data.QqqVolatility5),
                SafeFeature(data.QqqVolatility10),
                SafeFeature(data.MsftVolatility5),
                SafeFeature(data.MsftVolatility10),

                // Temporal & Cyclical features (20)
                // Day of Week Effects (5)
                data.IsMondayEffect,
                data.IsTuesdayEffect,
                data.IsWednesdayEffect,
                data.IsThursdayEffect,
                data.IsFridayEffect,

                // Week of Month Patterns (5)
                data.IsFirstWeekOfMonth,
                data.IsSecondWeekOfMonth,
                data.IsThirdWeekOfMonth,
                data.IsFourthWeekOfMonth,
                data.IsOptionsExpirationWeek,

                // Month & Quarter Effects (4)
                data.IsJanuaryEffect,
                data.IsQuarterStart,
                data.IsQuarterEnd,
                data.IsYearEnd,

                // Holiday & Cycle Proximity (6)
                Math.Min(data.DaysToMarketHoliday / 10.0, 1.0),
                Math.Min(data.DaysFromMarketHoliday / 10.0, 1.0),
                Math.Min(data.DaysIntoQuarter / 90.0, 1.0),
                Math.Min(data.DaysUntilQuarterEnd / 90.0, 1.0),
                data.QuarterProgress,
                data.YearProgress
            };

            return features.Count != 62
                ? throw new InvalidOperationException($"CRITICAL: Feature extraction error - expected exactly 62 features, got {features.Count}")
                : features.ToArray();
        }

        private static double SafeFeature(double value)
        {
            // Handle NaN, Infinity, and extreme values
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                return 0.0;
            }

            // Clamp extreme values
            return Math.Max(-1000, Math.Min(1000, value));
        }

        private void LogDataStatistics(string name, double[] data)
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

        private void LogFeatureStatistics(string name, double[,] features)
        {
            int rows = features.GetLength(0);
            int cols = features.GetLength(1);

            Console.WriteLine($"📊 {name} Matrix: {rows}x{cols}");

            for (int col = 0; col < Math.Min(cols, 8); col++)
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