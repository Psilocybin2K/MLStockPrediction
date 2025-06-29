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
        protected readonly InferenceEngine _engine;
        protected Gaussian[] _lowWeights;
        protected Gaussian _lowBias;
        protected Gamma _lowPrecision;
        protected Gaussian[] _highWeights;
        protected Gaussian _highBias;
        protected Gamma _highPrecision;
        protected bool _isTrained = false;

        // Fixed: Store normalization parameters separately for features and targets
        protected double[] _featureMeans;
        protected double[] _featureStds;
        protected double _lowTargetMean = 0;
        protected double _lowTargetStd = 1;
        protected double _highTargetMean = 0;
        protected double _highTargetStd = 1;

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

            // Log raw data statistics
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

        protected double[,] NormalizeFeatures(double[,] features)
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
                    // Clip extreme values to [-3, 3] to prevent outliers
                    normalized[row, col] = Math.Max(-3.0, Math.Min(3.0, normalizedValue));
                }
            }

            return normalized;
        }

        protected double[] NormalizeTargets(double[] targets, bool isLow)
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

        protected void TrainPriceModel(double[,] features, double[] targets, bool isLowModel)
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
                // Clip extreme values to [-3, 3] to prevent outliers
                normalizedFeatureVector[i] = Math.Max(-3.0, Math.Min(3.0, normalizedValue));
            }
            Console.WriteLine($"Normalized feature vector: [{string.Join(", ", normalizedFeatureVector.Select(x => $"{x:F4}"))}]");

            // Predict Low
            double lowPredictionNorm = this._lowBias.GetMean();
            for (int i = 0; i < normalizedFeatureVector.Length && i < this._lowWeights.Length; i++)
            {
                lowPredictionNorm += normalizedFeatureVector[i] * this._lowWeights[i].GetMean();
            }
            double lowPrediction = (lowPredictionNorm * this._lowTargetStd) + this._lowTargetMean;

            Console.WriteLine($"Low prediction: normalized={lowPredictionNorm:F4}, denormalized={lowPrediction:F2}");

            // Predict High
            double highPredictionNorm = this._highBias.GetMean();
            for (int i = 0; i < normalizedFeatureVector.Length && i < this._highWeights.Length; i++)
            {
                highPredictionNorm += normalizedFeatureVector[i] * this._highWeights[i].GetMean();
            }
            double highPrediction = (highPredictionNorm * this._highTargetStd) + this._highTargetMean;

            Console.WriteLine($"High prediction: normalized={highPredictionNorm:F4}, denormalized={highPrediction:F2}");

            // Ensure high >= low
            if (highPrediction < lowPrediction)
            {
                Console.WriteLine($"⚠️  Swapping predictions: High ({highPrediction:F2}) < Low ({lowPrediction:F2})");
                double temp = highPrediction;
                highPrediction = lowPrediction;
                lowPrediction = temp;
            }

            return (lowPrediction, highPrediction);
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
                matrix[i, 2] = Math.Log(1 + item.DowVolume); // Log transform for volume
                matrix[i, 3] = item.QqqReturn;
                matrix[i, 4] = item.QqqVolatility;
                matrix[i, 5] = Math.Log(1 + item.QqqVolume); // Log transform for volume
                matrix[i, 6] = item.MsftReturn;
                matrix[i, 7] = item.MsftVolatility; // Use actual volatility instead of correlation
                matrix[i, 8] = Math.Log(1 + item.MsftVolume); // Log transform for volume
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

        protected virtual void LogDataStatistics(string name, double[] data)
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

        protected virtual void LogFeatureStatistics(string name, double[,] features)
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

                Console.WriteLine($"   Feature[{col}]: Mean={mean:F4}, Min={min:F4}, Max={max:F4}");
            }
        }
    }


}