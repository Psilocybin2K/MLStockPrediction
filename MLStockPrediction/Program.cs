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
            Console.WriteLine("🚀 Starting Enhanced Bayesian Stock Prediction with Technical Features");

            StockDataLoader loader = new StockDataLoader();
            EnhancedMarketFeatureEngine featureEngine = new EnhancedMarketFeatureEngine();
            EnhancedBayesianStockModel enhancedModel = new EnhancedBayesianStockModel();
            StockModelEvaluator evaluator = new StockModelEvaluator();

            try
            {
                // Load data
                Console.WriteLine("📥 Loading stock data...");
                Dictionary<string, List<StockData>> allStockData = await loader.LoadAllStockDataAsync();
                loader.DisplayStockSummary(allStockData);

                // Create enhanced market features
                List<EnhancedMarketFeatures> enhancedFeatures = featureEngine.CreateMarketFeatures(allStockData);
                Console.WriteLine($"\n=== Enhanced Market Features Created ===");
                Console.WriteLine($"Enhanced feature records: {enhancedFeatures.Count}");
                Console.WriteLine($"Feature dimensions: 42 (9 original + 33 technical indicators)");

                if (enhancedFeatures.Count > 10)
                {
                    // Rolling time-series cross-validation instead of simple split
                    int testSize = Math.Min(20, enhancedFeatures.Count / 5);
                    int trainSize = enhancedFeatures.Count - testSize;

                    List<EnhancedMarketFeatures> trainData = enhancedFeatures.Take(trainSize).ToList();
                    List<EnhancedMarketFeatures> testData = enhancedFeatures.Skip(trainSize).ToList();

                    Console.WriteLine($"📊 Time-Series Split: {trainData.Count} train, {testData.Count} test");
                    Console.WriteLine($"📅 Training Period: {trainData.First().Date:yyyy-MM-dd} to {trainData.Last().Date:yyyy-MM-dd}");
                    Console.WriteLine($"📅 Testing Period: {testData.First().Date:yyyy-MM-dd} to {testData.Last().Date:yyyy-MM-dd}");

                    // Perform rolling time-series cross-validation with enhanced features
                    Console.WriteLine("\n🔄 Performing Enhanced Rolling Time-Series Cross-Validation...");
                    (double, double, double) validationResults = PerformEnhancedRollingValidation(enhancedFeatures, enhancedModel, 5);

                    Console.WriteLine($"📊 Enhanced Cross-Validation Results:");
                    Console.WriteLine($"   Average MAPE (Low): {validationResults.Item1:F2}%");
                    Console.WriteLine($"   Average MAPE (High): {validationResults.Item2:F2}%");
                    Console.WriteLine($"   Average Directional Accuracy: {validationResults.Item3:F1}%");

                    // Train enhanced Bayesian model on expanded training data
                    Console.WriteLine("\n🎯 Training Final Enhanced Model on Technical Features...");
                    enhancedModel.Train(trainData);
                    Console.WriteLine($"✅ Enhanced Bayesian model trained on {trainData.Count} samples with 42 features");

                    // Evaluate enhanced model
                    Console.WriteLine("\n🔍 Evaluating Enhanced Model Performance...");
                    StockEvaluationReport evaluationReport = evaluator.EvaluateEnhancedModel(enhancedModel, testData);
                    evaluator.PrintEvaluationReport(evaluationReport);

                    // Make prediction on latest data
                    EnhancedMarketFeatures latest = enhancedFeatures.Last();
                    Console.WriteLine($"\n🔮 Making Final Enhanced Prediction for {latest.Date:yyyy-MM-dd}...");
                    (double predictedLow, double predictedHigh) = enhancedModel.Predict(latest);

                    Console.WriteLine($"\n=== Enhanced MSFT Prediction for {latest.Date:yyyy-MM-dd} ===");
                    Console.WriteLine($"Predicted Low: ${predictedLow:F2}");
                    Console.WriteLine($"Predicted High: ${predictedHigh:F2}");
                    Console.WriteLine($"Actual Low: ${latest.MsftLow:F2}");
                    Console.WriteLine($"Actual High: ${latest.MsftHigh:F2}");

                    double lowError = Math.Abs(latest.MsftLow - predictedLow);
                    double highError = Math.Abs(latest.MsftHigh - predictedHigh);
                    double lowErrorPercent = (lowError / latest.MsftLow) * 100;
                    double highErrorPercent = (highError / latest.MsftHigh) * 100;

                    Console.WriteLine($"Enhanced Prediction Errors: Low={lowErrorPercent:F2}%, High={highErrorPercent:F2}%");

                    // Display key technical indicators for latest prediction
                    Console.WriteLine($"\n📊 Key Technical Indicators for Latest Prediction:");
                    Console.WriteLine($"   MSFT SMA20: ${latest.MsftSMA20:F2}");
                    Console.WriteLine($"   MSFT EMA Ratio (20-day): {latest.MsftEMAR20:F4}");
                    Console.WriteLine($"   MSFT Price Position: {latest.MsftPricePosition:F2} (0=low, 1=high in 20-day range)");
                    Console.WriteLine($"   MSFT ROC (10-day): {latest.MsftROC10:F4}");
                    Console.WriteLine($"   MSFT Volatility (20-day): {latest.MsftVolatility20:F4}");
                    Console.WriteLine($"   MSFT ATR: ${latest.MsftATR:F2}");
                    Console.WriteLine($"   MSFT Bollinger Position: {latest.MsftBBPosition:F2} (0=center, ±1=bands)");
                }
                else
                {
                    Console.WriteLine($"❌ Insufficient data for training. Need at least 10 samples, got {enhancedFeatures.Count}");
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

        private static (double, double, double) PerformEnhancedRollingValidation(
            List<EnhancedMarketFeatures> allData,
            EnhancedBayesianStockModel model,
            int numFolds)
        {
            List<double> lowMAPEs = new List<double>();
            List<double> highMAPEs = new List<double>();
            List<double> directionalAccuracies = new List<double>();

            int foldSize = allData.Count / (numFolds + 1);

            for (int fold = 0; fold < numFolds; fold++)
            {
                int trainStart = 0;
                int trainEnd = foldSize * (fold + 1);
                int testStart = trainEnd;
                int testEnd = Math.Min(testStart + foldSize / 2, allData.Count);

                if (testEnd <= testStart)
                {
                    break;
                }

                List<EnhancedMarketFeatures> foldTrainData = allData.GetRange(trainStart, trainEnd - trainStart);
                List<EnhancedMarketFeatures> foldTestData = allData.GetRange(testStart, testEnd - testStart);

                Console.WriteLine($"   Enhanced Fold {fold + 1}: Train[{trainStart}-{trainEnd}] Test[{testStart}-{testEnd}]");

                EnhancedBayesianStockModel foldModel = new EnhancedBayesianStockModel();
                foldModel.Train(foldTrainData);

                List<double> foldLowErrors = new List<double>();
                List<double> foldHighErrors = new List<double>();
                int correctDirections = 0;

                for (int i = 0; i < foldTestData.Count; i++)
                {
                    (double predLow, double predHigh) = foldModel.Predict(foldTestData[i]);

                    double lowError = Math.Abs((foldTestData[i].MsftLow - predLow) / foldTestData[i].MsftLow) * 100;
                    double highError = Math.Abs((foldTestData[i].MsftHigh - predHigh) / foldTestData[i].MsftHigh) * 100;

                    foldLowErrors.Add(lowError);
                    foldHighErrors.Add(highError);

                    if (i > 0)
                    {
                        int actualDirection = Math.Sign(foldTestData[i].MsftLow - foldTestData[i - 1].MsftLow);
                        int predDirection = Math.Sign(predLow - foldModel.Predict(foldTestData[i - 1]).Low);
                        if (actualDirection == predDirection)
                        {
                            correctDirections++;
                        }
                    }
                }

                if (foldLowErrors.Any())
                {
                    lowMAPEs.Add(foldLowErrors.Average());
                    highMAPEs.Add(foldHighErrors.Average());
                    directionalAccuracies.Add(correctDirections / (double)(foldTestData.Count - 1) * 100);
                }
            }

            return (
                lowMAPEs.Any() ? lowMAPEs.Average() : 0,
                highMAPEs.Any() ? highMAPEs.Average() : 0,
                directionalAccuracies.Any() ? directionalAccuracies.Average() : 0
            );
        }
    }
}