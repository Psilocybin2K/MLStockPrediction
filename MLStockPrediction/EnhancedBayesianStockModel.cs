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

        public EnhancedBayesianStockModel()
        {
            this._enhancedEngine = new InferenceEngine();
        }

        public void Train(List<EnhancedMarketFeatures> trainingData)
        {
            if (trainingData.Count < 2)
            {
                Console.WriteLine("⚠️  Insufficient enhanced training data");
                return;
            }

            Console.WriteLine($"🔧 Training enhanced model with {trainingData.Count} samples...");

            // Extract features and targets
            double[,] rawFeatures = this.ExtractEnhancedFeatureMatrix(trainingData);
            double[] lowTargets = trainingData.Select(x => x.MsftLow).ToArray();
            double[] highTargets = trainingData.Select(x => x.MsftHigh).ToArray();

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
            Console.WriteLine("🎯 Training Enhanced Low Price Model...");
            this.TrainEnhancedPriceModel(normalizedFeatures, normalizedLowTargets, true);

            Console.WriteLine("🎯 Training Enhanced High Price Model...");
            this.TrainEnhancedPriceModel(normalizedFeatures, normalizedHighTargets, false);

            Console.WriteLine("✅ Enhanced model training completed");
        }

        public (double Low, double High) Predict(EnhancedMarketFeatures marketData)
        {
            if (!this._enhancedIsTrained)
            {
                throw new InvalidOperationException("Enhanced model must be trained first");
            }

            double[] rawFeatureVector = this.ExtractEnhancedFeatureVector(marketData);
            Console.WriteLine($"Enhanced feature vector length: {rawFeatureVector.Length}");

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

            Console.WriteLine($"Training Enhanced {(isLowModel ? "Low" : "High")} model: {n} samples, {numFeatures} features");

            // Define ranges
            Range dataRange = new Range(n);
            Range featureRange = new Range(numFeatures);

            // Use more conservative priors for larger feature space
            VariableArray<double> weights = Variable.Array<double>(featureRange);
            weights[featureRange] = Variable.GaussianFromMeanAndVariance(0, 0.001).ForEach(featureRange);

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
            Console.WriteLine("Running enhanced inference...");
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
            int numFeatures = 42; // 9 original + 33 enhanced features
            double[,] matrix = new double[n, numFeatures];

            for (int i = 0; i < n; i++)
            {
                EnhancedMarketFeatures item = data[i];
                double[] featureVector = this.ExtractEnhancedFeatureVector(item);

                // Copy feature vector to matrix row
                for (int j = 0; j < Math.Min(featureVector.Length, numFeatures); j++)
                {
                    matrix[i, j] = featureVector[j];
                }
            }

            return matrix;
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

            // Enhanced features (33)
            // Moving Averages (9) - Handle potential NaN/Infinity
            this.SafeFeature(data.DowSMA5),
            this.SafeFeature(data.DowSMA10),
            this.SafeFeature(data.DowSMA20),
            this.SafeFeature(data.QqqSMA5),
            this.SafeFeature(data.QqqSMA10),
            this.SafeFeature(data.QqqSMA20),
            this.SafeFeature(data.MsftSMA5),
            this.SafeFeature(data.MsftSMA10),
            this.SafeFeature(data.MsftSMA20),

            // EMA Ratios (9)
            this.SafeFeature(data.DowEMAR5),
            this.SafeFeature(data.DowEMAR10),
            this.SafeFeature(data.DowEMAR20),
            this.SafeFeature(data.QqqEMAR5),
            this.SafeFeature(data.QqqEMAR10),
            this.SafeFeature(data.QqqEMAR20),
            this.SafeFeature(data.MsftEMAR5),
            this.SafeFeature(data.MsftEMAR10),
            this.SafeFeature(data.MsftEMAR20),

            // Price Positions (3)
            this.SafeFeature(data.DowPricePosition),
            this.SafeFeature(data.QqqPricePosition),
            this.SafeFeature(data.MsftPricePosition),

            // Rate of Change (6)
            this.SafeFeature(data.DowROC5),
            this.SafeFeature(data.DowROC10),
            this.SafeFeature(data.QqqROC5),
            this.SafeFeature(data.QqqROC10),
            this.SafeFeature(data.MsftROC5),
            this.SafeFeature(data.MsftROC10),

            // Rolling Volatility (9)
            this.SafeFeature(data.DowVolatility5),
            this.SafeFeature(data.DowVolatility10),
            this.SafeFeature(data.DowVolatility20),
            this.SafeFeature(data.QqqVolatility5),
            this.SafeFeature(data.QqqVolatility10),
            this.SafeFeature(data.QqqVolatility20),
            this.SafeFeature(data.MsftVolatility5),
            this.SafeFeature(data.MsftVolatility10),
            this.SafeFeature(data.MsftVolatility20),

            // ATR and ratios (6)
            this.SafeFeature(data.DowATR),
            this.SafeFeature(data.QqqATR),
            this.SafeFeature(data.MsftATR),
            this.SafeFeature(data.DowVolatilityRatio),
            this.SafeFeature(data.QqqVolatilityRatio),
            this.SafeFeature(data.MsftVolatilityRatio),

            // Bollinger Band Positions (3)
            this.SafeFeature(data.DowBBPosition),
            this.SafeFeature(data.QqqBBPosition),
            this.SafeFeature(data.MsftBBPosition)
        };

            return features.ToArray();
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

            for (int col = 0; col < Math.Min(cols, 5); col++) // Log first 5 features
            {
                List<double> columnData = new List<double>();
                for (int row = 0; row < rows; row++)
                {
                    columnData.Add(features[row, col]);
                }

                double mean = columnData.Average();
                double min = columnData.Min();
                double max = columnData.Max();

                Console.WriteLine($"   Enhanced Feature[{col}]: Mean={mean:F4}, Min={min:F4}, Max={max:F4}");
            }
        }
    }
}