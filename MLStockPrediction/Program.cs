namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;

    using MLStockPrediction.Models;

    public class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("🚀 Starting Stacking Ensemble Stock Prediction");

            StockDataLoader loader = new StockDataLoader();
            EnhancedFeatureEngine featureEngine = new EnhancedFeatureEngine();
            EnsembleStockModel ensembleModel = new EnsembleStockModel();

            try
            {
                // Load data
                Console.WriteLine("📥 Loading stock data...");
                Dictionary<string, List<StockData>> allStockData = await loader.LoadAllStockDataAsync();
                loader.DisplayStockSummary(allStockData);

                // Create enhanced market features for ensemble
                List<EnhancedMarketFeatures> enhancedFeatures = featureEngine.CreateEnhancedMarketFeaturesForEnsemble(allStockData);
                Console.WriteLine($"\n=== Enhanced Ensemble Features Created ===");
                Console.WriteLine($"Enhanced feature records: {enhancedFeatures.Count}");

                if (enhancedFeatures.Count > 60)
                {
                    // Traditional time-series split for final evaluation
                    int testSize = Math.Min(20, enhancedFeatures.Count / 5);
                    int trainSize = enhancedFeatures.Count - testSize;

                    List<EnhancedMarketFeatures> trainData = enhancedFeatures.Take(trainSize).ToList();
                    List<EnhancedMarketFeatures> testData = enhancedFeatures.Skip(trainSize).ToList();

                    Console.WriteLine($"\n📊 Final Ensemble Training Split: {trainData.Count} train, {testData.Count} test");
                    Console.WriteLine($"📅 Training Period: {trainData.First().Date:yyyy-MM-dd} to {trainData.Last().Date:yyyy-MM-dd}");
                    Console.WriteLine($"📅 Testing Period: {testData.First().Date:yyyy-MM-dd} to {testData.Last().Date:yyyy-MM-dd}");

                    // Train ensemble model
                    Console.WriteLine("\n🎯 Training Stacking Ensemble Model...");
                    ensembleModel.Train(trainData);
                    Console.WriteLine($"✅ Stacking ensemble model trained on {trainData.Count} samples.");

                    // Evaluate the ensemble model
                    Console.WriteLine("\n🔍 Evaluating Stacking Ensemble Performance...");
                    List<EnsemblePredictionResult> predictions = new List<EnsemblePredictionResult>();
                    foreach (EnhancedMarketFeatures sample in testData)
                    {
                        EnsemblePredictionResult result = ensembleModel.Predict(sample);
                        predictions.Add(result);
                    }

                    double mapeLow = CalculateEnsembleMAPE(predictions, testData, true);
                    double mapeHigh = CalculateEnsembleMAPE(predictions, testData, false);

                    Console.WriteLine("\n📊 STACKING ENSEMBLE MODEL PERFORMANCE:");
                    Console.WriteLine($"   MAPE: Low={mapeLow:F2}%, High={mapeHigh:F2}%");

                    // Make final prediction on latest data
                    EnhancedMarketFeatures latest = enhancedFeatures.Last();
                    Console.WriteLine($"\n🔮 Making Final Ensemble Prediction for {latest.Date:yyyy-MM-dd}...");
                    EnsemblePredictionResult finalResult = ensembleModel.Predict(latest);

                    Console.WriteLine($"\n=== STACKING ENSEMBLE PREDICTION for {latest.Date:yyyy-MM-dd} ===");
                    Console.WriteLine($"Actual: Low=${latest.MsftLow:F2}, High=${latest.MsftHigh:F2}");
                    Console.WriteLine($"🎯 Final Stacked Prediction: Low=${finalResult.FinalLow:F2}, High=${finalResult.FinalHigh:F2}");
                    Console.WriteLine($"🎯 Ensemble Errors: Low={Math.Abs(latest.MsftLow - finalResult.FinalLow) / latest.MsftLow * 100:F2}%, High={Math.Abs(latest.MsftHigh - finalResult.FinalHigh) / latest.MsftHigh * 100:F2}%");
                }
                else
                {
                    Console.WriteLine($"❌ Insufficient data for ensemble training. Need at least 60 samples, got {enhancedFeatures.Count}");
                }
            }
            catch (FileNotFoundException ex)
            {
                Console.WriteLine($"❌ File not found: {ex.FileName}");
                Console.WriteLine("Please ensure DOW.csv, QQQ.csv, and MSFT.csv are in the application directory.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }

            Debugger.Break();
        }

        private static double CalculateEnsembleMAPE(List<EnsemblePredictionResult> results, List<EnhancedMarketFeatures> actuals, bool isLow)
        {
            if (results.Count == 0 || actuals.Count == 0)
            {
                return 0.0;
            }

            double totalError = 0;
            int count = 0;

            for (int i = 0; i < Math.Min(results.Count, actuals.Count); i++)
            {
                EnsemblePredictionResult result = results[i];
                EnhancedMarketFeatures actual = actuals[i];

                if (result.Date != actual.Date)
                {
                    continue;
                }

                double predicted = isLow ? result.FinalLow : result.FinalHigh;
                double actualValue = isLow ? actual.MsftLow : actual.MsftHigh;

                if (actualValue > 0)
                {
                    totalError += Math.Abs((actualValue - predicted) / actualValue) * 100;
                    count++;
                }
            }

            return count > 0 ? totalError / count : 0.0;
        }
    }
}