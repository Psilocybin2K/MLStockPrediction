namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MLStockPrediction.Models;

    public class EnsembleStockModel
    {
        private readonly BayesianStockModel _bayesianModel;
        private readonly LightGbmStockPredictor _lightGbmModel;
        private readonly EnsembleWeights _weights;
        private readonly List<EnsemblePredictionResult> _predictionHistory;
        private readonly List<ModelPerformanceMetrics> _performanceHistory;
        private bool _isTrained = false;

        // Performance tracking parameters
        private const int PerformanceWindowSize = 20;
        private const int WeightUpdateFrequency = 5;
        private const double MinWeight = 0.1;
        private const double MaxWeight = 0.8;

        public EnsembleStockModel()
        {
            this._bayesianModel = new BayesianStockModel();
            this._lightGbmModel = new LightGbmStockPredictor();
            this._weights = new EnsembleWeights();
            this._predictionHistory = new List<EnsemblePredictionResult>();
            this._performanceHistory = new List<ModelPerformanceMetrics>();
        }

        public void Train(List<EnhancedMarketFeatures> trainingData)
        {
            if (trainingData.Count < 20)
            {
                Console.WriteLine("⚠️  Insufficient data for ensemble training");
                return;
            }

            Console.WriteLine($"🚀 Training Ensemble Model on {trainingData.Count} samples...");

            // Train both models
            Console.WriteLine("📊 Training Bayesian Model...");
            this._bayesianModel.Train(trainingData);

            Console.WriteLine("🌟 Training LightGBM Models...");
            this._lightGbmModel.Train(trainingData);

            // Initialize weights with cross-validation
            this.InitializeWeightsWithCrossValidation(trainingData);

            this._isTrained = true;
            Console.WriteLine("✅ Ensemble model training completed");
        }

        public EnsemblePredictionResult Predict(EnhancedMarketFeatures marketData)
        {
            if (!this._isTrained)
            {
                throw new InvalidOperationException("Ensemble model must be trained first");
            }

            Console.WriteLine($"\n🔮 Ensemble Prediction for {marketData.Date:yyyy-MM-dd}");

            // Get predictions from both models
            (double bayesianLow, double bayesianHigh) = this._bayesianModel.Predict(marketData);
            (double lightGbmLow, double lightGbmHigh, double lightGbmRange) = this._lightGbmModel.Predict(marketData);

            // Detect market regime
            string marketRegime = this.DetectMarketRegime(marketData);

            // Adjust weights based on market regime
            EnsembleWeights adjustedWeights = this.AdjustWeightsForMarketRegime(this._weights, marketRegime);

            // Combine predictions
            double finalLow = this.CombineLowPredictions(bayesianLow, lightGbmLow, adjustedWeights);
            double finalHigh = this.CombineHighPredictions(bayesianHigh, lightGbmHigh, adjustedWeights);

            // Apply range validation
            (finalLow, finalHigh) = this.ValidateWithRange(finalLow, finalHigh, lightGbmRange, adjustedWeights);

            // Calculate Bayesian confidence (distance from prediction bounds)
            double bayesianConfidence = this.CalculateBayesianConfidence(bayesianLow, bayesianHigh);

            EnsemblePredictionResult result = new EnsemblePredictionResult
            {
                Date = marketData.Date,
                BayesianLow = bayesianLow,
                BayesianHigh = bayesianHigh,
                LightGbmLow = lightGbmLow,
                LightGbmHigh = lightGbmHigh,
                LightGbmRange = lightGbmRange,
                FinalLow = finalLow,
                FinalHigh = finalHigh,
                Weights = this.CloneWeights(adjustedWeights),
                BayesianConfidence = bayesianConfidence,
                MarketRegime = marketRegime
            };

            this._predictionHistory.Add(result);

            Console.WriteLine($"   Bayesian: Low=${bayesianLow:F2}, High=${bayesianHigh:F2}");
            Console.WriteLine($"   LightGBM: Low=${lightGbmLow:F2}, High=${lightGbmHigh:F2}, Range=${lightGbmRange:F2}");
            Console.WriteLine($"   Final: Low=${finalLow:F2}, High=${finalHigh:F2}");
            Console.WriteLine($"   Regime: {marketRegime}, Confidence: {bayesianConfidence:F3}");
            Console.WriteLine($"   Weights: Bayesian={adjustedWeights.BayesianWeight:F2}, LightGBM={adjustedWeights.LightGbmLowWeight:F2}");

            return result;
        }

        public void UpdatePerformance(EnhancedMarketFeatures actualData)
        {
            if (this._predictionHistory.Count == 0)
            {
                return;
            }

            EnsemblePredictionResult lastPrediction = this._predictionHistory.Last();
            if (lastPrediction.Date != actualData.Date)
            {
                return;
            }

            // Calculate errors
            double lowError = Math.Abs(actualData.MsftLow - lastPrediction.FinalLow);
            double highError = Math.Abs(actualData.MsftHigh - lastPrediction.FinalHigh);
            double lowPercentError = lowError / actualData.MsftLow * 100;
            double highPercentError = highError / actualData.MsftHigh * 100;

            // Calculate individual model errors for weight updates
            double bayesianLowError = Math.Abs(actualData.MsftLow - lastPrediction.BayesianLow);
            double bayesianHighError = Math.Abs(actualData.MsftHigh - lastPrediction.BayesianHigh);
            double lightGbmLowError = Math.Abs(actualData.MsftLow - lastPrediction.LightGbmLow);
            double lightGbmHighError = Math.Abs(actualData.MsftHigh - lastPrediction.LightGbmHigh);

            Console.WriteLine($"📊 Performance Update for {actualData.Date:yyyy-MM-dd}:");
            Console.WriteLine($"   Actual: Low=${actualData.MsftLow:F2}, High=${actualData.MsftHigh:F2}");
            Console.WriteLine($"   Ensemble Errors: Low={lowPercentError:F2}%, High={highPercentError:F2}%");

            // Update weights if we have enough history
            if (this._predictionHistory.Count % WeightUpdateFrequency == 0 && this._predictionHistory.Count >= PerformanceWindowSize)
            {
                this.UpdateEnsembleWeights();
            }
        }

        public void EnableCalibration(bool enable = true)
        {
            this._bayesianModel.EnableCalibration(enable);
            Console.WriteLine($"📊 Bayesian calibration {(enable ? "enabled" : "disabled")} in ensemble");
        }

        private void InitializeWeightsWithCrossValidation(List<EnhancedMarketFeatures> data)
        {
            Console.WriteLine("⚖️  Initializing ensemble weights with cross-validation...");

            // Simple 80/20 split for weight initialization
            int splitIndex = (int)(data.Count * 0.8);
            List<EnhancedMarketFeatures> trainData = data.Take(splitIndex).ToList();
            List<EnhancedMarketFeatures> validData = data.Skip(splitIndex).ToList();

            if (validData.Count < 5)
            {
                Console.WriteLine("⚠️  Using default weights due to insufficient validation data");
                return;
            }

            // Quick validation on small subset
            double bayesianMAPE = 0, lightGbmMAPE = 0;
            int validCount = 0;

            foreach (EnhancedMarketFeatures? sample in validData)
            {
                try
                {
                    (double bLow, double bHigh) = this._bayesianModel.Predict(sample);
                    (double lLow, double lHigh, _) = this._lightGbmModel.Predict(sample);

                    double bError = (Math.Abs(sample.MsftLow - bLow) / sample.MsftLow +
                                   Math.Abs(sample.MsftHigh - bHigh) / sample.MsftHigh) * 50;
                    double lError = (Math.Abs(sample.MsftLow - lLow) / sample.MsftLow +
                                   Math.Abs(sample.MsftHigh - lHigh) / sample.MsftHigh) * 50;

                    bayesianMAPE += bError;
                    lightGbmMAPE += lError;
                    validCount++;
                }
                catch
                {
                    // Skip problematic samples
                    continue;
                }
            }

            if (validCount > 0)
            {
                bayesianMAPE /= validCount;
                lightGbmMAPE /= validCount;

                // Calculate weights inversely proportional to error
                double totalInverseError = (1.0 / bayesianMAPE) + (1.0 / lightGbmMAPE);
                double bayesianWeight = (1.0 / bayesianMAPE) / totalInverseError;
                double lightGbmWeight = (1.0 / lightGbmMAPE) / totalInverseError;

                // Apply constraints
                bayesianWeight = Math.Max(MinWeight, Math.Min(MaxWeight, bayesianWeight));
                lightGbmWeight = Math.Max(MinWeight, Math.Min(MaxWeight, lightGbmWeight));

                this._weights.BayesianWeight = bayesianWeight;
                this._weights.LightGbmLowWeight = lightGbmWeight;
                this._weights.LightGbmHighWeight = lightGbmWeight;

                Console.WriteLine($"✅ Initialized weights: Bayesian={bayesianWeight:F3}, LightGBM={lightGbmWeight:F3}");
                Console.WriteLine($"   Based on validation MAPE: Bayesian={bayesianMAPE:F2}%, LightGBM={lightGbmMAPE:F2}%");
            }
        }

        private string DetectMarketRegime(EnhancedMarketFeatures marketData)
        {
            // Simple regime detection based on volatility and momentum
            double volatility = marketData.MsftVolatility10;
            double momentum = marketData.MsftROC10;
            double volume = marketData.MsftVolume;

            return volatility > 0.03
                ? "High Volatility"
                : momentum > 0.02 ? "Bull Trend" : momentum < -0.02 ? "Bear Trend" : volume > 1.5 ? "High Volume Sideways" : "Normal";
        }

        private EnsembleWeights AdjustWeightsForMarketRegime(EnsembleWeights baseWeights, string regime)
        {
            EnsembleWeights adjusted = this.CloneWeights(baseWeights);

            switch (regime)
            {
                case "High Volatility":
                    // Favor Bayesian for uncertainty quantification
                    adjusted.BayesianWeight = Math.Min(MaxWeight, adjusted.BayesianWeight * 1.2);
                    adjusted.LightGbmLowWeight = Math.Max(MinWeight, adjusted.LightGbmLowWeight * 0.9);
                    adjusted.LightGbmHighWeight = Math.Max(MinWeight, adjusted.LightGbmHighWeight * 0.9);
                    break;

                case "Bull Trend":
                case "Bear Trend":
                    // Favor LightGBM for trend following
                    adjusted.BayesianWeight = Math.Max(MinWeight, adjusted.BayesianWeight * 0.9);
                    adjusted.LightGbmLowWeight = Math.Min(MaxWeight, adjusted.LightGbmLowWeight * 1.1);
                    adjusted.LightGbmHighWeight = Math.Min(MaxWeight, adjusted.LightGbmHighWeight * 1.1);
                    break;

                case "High Volume Sideways":
                    // Balanced approach
                    break;

                default: // Normal
                    // Use base weights
                    break;
            }

            // Normalize weights
            this.NormalizeWeights(adjusted);
            return adjusted;
        }

        private double CombineLowPredictions(double bayesian, double lightGbm, EnsembleWeights weights)
        {
            return (bayesian * weights.BayesianWeight) + (lightGbm * weights.LightGbmLowWeight);
        }

        private double CombineHighPredictions(double bayesian, double lightGbm, EnsembleWeights weights)
        {
            return (bayesian * weights.BayesianWeight) + (lightGbm * weights.LightGbmHighWeight);
        }

        private (double Low, double High) ValidateWithRange(double low, double high, double predictedRange, EnsembleWeights weights)
        {
            double currentRange = high - low;
            double rangeDifference = Math.Abs(currentRange - predictedRange);

            // If range difference is significant, apply adjustment
            if (rangeDifference > predictedRange * 0.2) // 20% threshold
            {
                double adjustment = (predictedRange - currentRange) * weights.RangeAdjustmentWeight;
                double midPoint = (low + high) / 2;

                // Adjust around midpoint
                low = midPoint - (predictedRange / 2) * (1 - weights.RangeAdjustmentWeight) - (currentRange / 2) * weights.RangeAdjustmentWeight;
                high = midPoint + (predictedRange / 2) * (1 - weights.RangeAdjustmentWeight) + (currentRange / 2) * weights.RangeAdjustmentWeight;
            }

            // Ensure high > low
            if (high <= low)
            {
                double midPoint = (low + high) / 2;
                low = midPoint - 0.01;
                high = midPoint + 0.01;
            }

            return (low, high);
        }

        private double CalculateBayesianConfidence(double low, double high)
        {
            double range = high - low;
            double midPoint = (high + low) / 2;

            // Confidence inversely related to prediction range
            // Normalize to 0-1 scale where smaller ranges = higher confidence
            return Math.Max(0.1, Math.Min(1.0, 1.0 / (1.0 + range / midPoint)));
        }

        private void UpdateEnsembleWeights()
        {
            Console.WriteLine("⚖️  Updating ensemble weights based on recent performance...");

            List<EnsemblePredictionResult> recentPredictions = this._predictionHistory.TakeLast(PerformanceWindowSize).ToList();
            if (recentPredictions.Count < PerformanceWindowSize)
            {
                return;
            }

            foreach (EnsemblePredictionResult? prediction in recentPredictions)
            {
                // We need the actual values to calculate performance
                // For now, we'll skip this update if we don't have actuals
                // This would be enhanced in a real implementation with actual data tracking
            }

            // Simple weight decay to prevent stagnation
            this._weights.BayesianWeight = Math.Max(MinWeight, this._weights.BayesianWeight * 0.95 + 0.05);
            this._weights.LightGbmLowWeight = Math.Max(MinWeight, this._weights.LightGbmLowWeight * 0.95 + 0.05);
            this._weights.LightGbmHighWeight = Math.Max(MinWeight, this._weights.LightGbmHighWeight * 0.95 + 0.05);

            this.NormalizeWeights(this._weights);

            this._weights.LastUpdated = DateTime.Now;
            this._weights.UpdateCount++;

            Console.WriteLine($"✅ Weights updated: Bayesian={this._weights.BayesianWeight:F3}, LightGBM={this._weights.LightGbmLowWeight:F3}");
        }

        private void NormalizeWeights(EnsembleWeights weights)
        {
            double totalWeight = weights.BayesianWeight +
                               Math.Max(weights.LightGbmLowWeight, weights.LightGbmHighWeight);

            if (totalWeight > 0)
            {
                double factor = 1.0 / totalWeight;
                weights.BayesianWeight *= factor;
                weights.LightGbmLowWeight *= factor;
                weights.LightGbmHighWeight *= factor;
            }
        }

        private EnsembleWeights CloneWeights(EnsembleWeights original)
        {
            return new EnsembleWeights
            {
                BayesianWeight = original.BayesianWeight,
                LightGbmLowWeight = original.LightGbmLowWeight,
                LightGbmHighWeight = original.LightGbmHighWeight,
                RangeAdjustmentWeight = original.RangeAdjustmentWeight,
                LastUpdated = original.LastUpdated,
                UpdateCount = original.UpdateCount
            };
        }

        public void PrintEnsembleStatistics()
        {
            Console.WriteLine("\n" + "=".PadRight(60, '='));
            Console.WriteLine("📊 ENSEMBLE MODEL STATISTICS");
            Console.WriteLine("=".PadRight(60, '='));

            Console.WriteLine($"\n🎯 CURRENT WEIGHTS:");
            Console.WriteLine($"   Bayesian Weight: {this._weights.BayesianWeight:F3}");
            Console.WriteLine($"   LightGBM Low Weight: {this._weights.LightGbmLowWeight:F3}");
            Console.WriteLine($"   LightGBM High Weight: {this._weights.LightGbmHighWeight:F3}");
            Console.WriteLine($"   Range Adjustment: {this._weights.RangeAdjustmentWeight:F3}");
            Console.WriteLine($"   Last Updated: {this._weights.LastUpdated:yyyy-MM-dd HH:mm}");
            Console.WriteLine($"   Update Count: {this._weights.UpdateCount}");

            if (this._predictionHistory.Count > 0)
            {
                Console.WriteLine($"\n📈 PREDICTION HISTORY:");
                Console.WriteLine($"   Total Predictions: {this._predictionHistory.Count}");

                List<EnsemblePredictionResult> recentPredictions = this._predictionHistory.TakeLast(5).ToList();
                Console.WriteLine($"   Recent Predictions:");

                foreach (EnsemblePredictionResult? pred in recentPredictions)
                {
                    Console.WriteLine($"     {pred.Date:MM/dd}: Low=${pred.FinalLow:F2}, High=${pred.FinalHigh:F2} ({pred.MarketRegime})");
                }

                var regimeDistribution = this._predictionHistory
                    .GroupBy(p => p.MarketRegime)
                    .Select(g => new { Regime = g.Key, Count = g.Count() })
                    .OrderByDescending(x => x.Count);

                Console.WriteLine($"\n🎭 MARKET REGIME DISTRIBUTION:");
                foreach (var regime in regimeDistribution)
                {
                    double percentage = (double)regime.Count / this._predictionHistory.Count * 100;
                    Console.WriteLine($"   {regime.Regime}: {regime.Count} ({percentage:F1}%)");
                }
            }

            Console.WriteLine("\n" + "=".PadRight(60, '='));
        }
    }
}