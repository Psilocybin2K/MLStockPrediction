namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;

    using MLStockPrediction.Evaluations;
    using MLStockPrediction.Models;

    public class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("🚀 Starting Enhanced Bayesian Stock Prediction with Unified Model");

            StockDataLoader loader = new StockDataLoader();
            MarketFeatureEngine featureEngine = new MarketFeatureEngine();
            BayesianStockModel model = new BayesianStockModel();
            StockModelEvaluator evaluator = new StockModelEvaluator();
            WalkForwardValidator walkForwardValidator = new WalkForwardValidator();

            try
            {
                // Load data
                Console.WriteLine("📥 Loading stock data...");
                Dictionary<string, List<StockData>> allStockData = await loader.LoadAllStockDataAsync();
                loader.DisplayStockSummary(allStockData);

                // Create enhanced market features
                List<EnhancedMarketFeatures> enhancedFeatures = featureEngine.CreateEnhancedMarketFeatures(allStockData);
                Console.WriteLine($"\n=== Enhanced Market Features Created ===");
                Console.WriteLine($"Enhanced feature records: {enhancedFeatures.Count}");
                Console.WriteLine($"Feature dimensions: 62 (9 original + 33 technical + 20 temporal) - UNIFIED ENGINE");

                if (enhancedFeatures.Count > 60)
                {
                    // Walk-Forward Validation
                    Console.WriteLine("\n🔄 Performing Walk-Forward Validation...");
                    WalkForwardValidationResult walkForwardResult = walkForwardValidator.ValidateModel(
                        enhancedFeatures,
                        initialTrainingSize: 50,
                        validationWindow: 10,
                        stepSize: 5);

                    walkForwardResult.PrintDetailedReport();

                    // Traditional time-series split for final evaluation
                    int testSize = Math.Min(20, enhancedFeatures.Count / 5);
                    int trainSize = enhancedFeatures.Count - testSize;

                    List<EnhancedMarketFeatures> trainData = enhancedFeatures.Take(trainSize).ToList();
                    List<EnhancedMarketFeatures> testData = enhancedFeatures.Skip(trainSize).ToList();

                    Console.WriteLine($"\n📊 Final Model Training Split: {trainData.Count} train, {testData.Count} test");
                    Console.WriteLine($"📅 Training Period: {trainData.First().Date:yyyy-MM-dd} to {trainData.Last().Date:yyyy-MM-dd}");
                    Console.WriteLine($"📅 Testing Period: {testData.First().Date:yyyy-MM-dd} to {testData.Last().Date:yyyy-MM-dd}");

                    // Train final unified Bayesian model
                    Console.WriteLine("\n🎯 Training Final Unified Model with Enhanced Features...");
                    model.Train(trainData);
                    Console.WriteLine($"✅ Unified Bayesian model trained on {trainData.Count} samples with exactly 62 features");

                    // Test both uncalibrated and calibrated models
                    Console.WriteLine("\n🔍 Evaluating Model Performance (Uncalibrated vs Calibrated)...");

                    // Test uncalibrated first
                    model.EnableCalibration(false);
                    StockEvaluationReport uncalibratedReport = evaluator.EvaluateModel(model, testData);
                    Console.WriteLine("\n📊 UNCALIBRATED MODEL PERFORMANCE:");
                    evaluator.PrintEvaluationReport(uncalibratedReport);

                    // Test calibrated
                    model.EnableCalibration(true);
                    StockEvaluationReport calibratedReport = evaluator.EvaluateModel(model, testData);
                    Console.WriteLine("\n📊 CALIBRATED MODEL PERFORMANCE:");
                    evaluator.PrintEvaluationReport(calibratedReport);

                    // Compare performance
                    Console.WriteLine("\n📈 CALIBRATION IMPACT SUMMARY:");
                    Console.WriteLine($"   Low MAPE Improvement: {uncalibratedReport.LowPriceMetrics.MAPE - calibratedReport.LowPriceMetrics.MAPE:F2} percentage points");
                    Console.WriteLine($"   High MAPE Improvement: {uncalibratedReport.HighPriceMetrics.MAPE - calibratedReport.HighPriceMetrics.MAPE:F2} percentage points");
                    Console.WriteLine($"   Low Directional Accuracy Change: {calibratedReport.DirectionalAccuracy.LowDirectionalAccuracy - uncalibratedReport.DirectionalAccuracy.LowDirectionalAccuracy:F1} percentage points");
                    Console.WriteLine($"   High Directional Accuracy Change: {calibratedReport.DirectionalAccuracy.HighDirectionalAccuracy - uncalibratedReport.DirectionalAccuracy.HighDirectionalAccuracy:F1} percentage points");

                    // Make prediction on latest data with both models
                    EnhancedMarketFeatures latest = enhancedFeatures.Last();

                    Console.WriteLine($"\n🔮 Making Final Predictions for {latest.Date:yyyy-MM-dd}...");

                    // Uncalibrated prediction
                    model.EnableCalibration(false);
                    (double uncalLow, double uncalHigh) = model.Predict(latest);

                    // Calibrated prediction
                    model.EnableCalibration(true);
                    (double calLow, double calHigh) = model.Predict(latest);

                    Console.WriteLine($"\n=== MSFT Prediction Comparison for {latest.Date:yyyy-MM-dd} ===");
                    Console.WriteLine($"Actual Low: ${latest.MsftLow:F2}, High: ${latest.MsftHigh:F2}");
                    Console.WriteLine($"");
                    Console.WriteLine($"Uncalibrated Prediction: Low=${uncalLow:F2}, High=${uncalHigh:F2}");
                    Console.WriteLine($"Uncalibrated Errors: Low={Math.Abs(latest.MsftLow - uncalLow) / latest.MsftLow * 100:F2}%, High={Math.Abs(latest.MsftHigh - uncalHigh) / latest.MsftHigh * 100:F2}%");
                    Console.WriteLine($"");
                    Console.WriteLine($"Calibrated Prediction: Low=${calLow:F2}, High=${calHigh:F2}");
                    Console.WriteLine($"Calibrated Errors: Low={Math.Abs(latest.MsftLow - calLow) / latest.MsftLow * 100:F2}%, High={Math.Abs(latest.MsftHigh - calHigh) / latest.MsftHigh * 100:F2}%");

                    // Display key technical indicators for latest prediction
                    Console.WriteLine($"\n📊 Key Technical Indicators for Latest Prediction:");
                    Console.WriteLine($"   MSFT SMA20: ${latest.MsftSMA20:F2}");
                    Console.WriteLine($"   MSFT EMA Ratio (20-day): {latest.MsftEMAR20:F4}");
                    Console.WriteLine($"   MSFT Price Position: {latest.MsftPricePosition:F2} (0=low, 1=high in 20-day range)");
                    Console.WriteLine($"   MSFT ROC (10-day): {latest.MsftROC10:F4}");
                    Console.WriteLine($"   MSFT Volatility (10-day): {latest.MsftVolatility10:F4}");
                    Console.WriteLine($"   MSFT ATR: ${latest.MsftATR:F2}");
                    Console.WriteLine($"   MSFT Bollinger Position: {latest.MsftBBPosition:F2} (0=center, ±1=bands)");

                    Console.WriteLine($"\n📅 Temporal Features for Latest Prediction:");
                    Console.WriteLine($"   Day of Week: {latest.Date.DayOfWeek}");
                    Console.WriteLine($"   Options Expiration Week: {latest.IsOptionsExpirationWeek == 1.0}");
                    Console.WriteLine($"   Quarter Progress: {latest.QuarterProgress:F2}");
                    Console.WriteLine($"   Year Progress: {latest.YearProgress:F2}");
                    Console.WriteLine($"   Days to Holiday: {latest.DaysToMarketHoliday}");
                    Console.WriteLine($"   Earnings Season: {latest.IsEarningsSeason == 1.0}");

                    // Demonstrate basic model capability as well
                    Console.WriteLine($"\n🔄 Testing Basic Model Capability...");

                    // Create basic features using the unified engine
                    List<MarketFeatures> basicFeatures = featureEngine.CreateMarketFeatures(allStockData);

                    List<MarketFeatures> basicTrainData = basicFeatures.Take(trainSize).ToList();
                    List<MarketFeatures> basicTestData = basicFeatures.Skip(trainSize).ToList();

                    // Train basic model
                    BayesianStockModel basicModel = new BayesianStockModel();
                    basicModel.Train(basicTrainData);

                    // Evaluate basic model
                    StockEvaluationReport basicReport = evaluator.EvaluateModel(basicModel, basicTestData);
                    Console.WriteLine("\n📊 BASIC MODEL PERFORMANCE (9 features):");
                    evaluator.PrintEvaluationReport(basicReport);

                    // Compare basic vs enhanced
                    Console.WriteLine("\n📈 ENHANCED vs BASIC MODEL COMPARISON:");
                    Console.WriteLine($"   Enhanced Model MAPE: Low={calibratedReport.LowPriceMetrics.MAPE:F2}%, High={calibratedReport.HighPriceMetrics.MAPE:F2}%");
                    Console.WriteLine($"   Basic Model MAPE: Low={basicReport.LowPriceMetrics.MAPE:F2}%, High={basicReport.HighPriceMetrics.MAPE:F2}%");
                    Console.WriteLine($"   Enhancement Improvement: Low={basicReport.LowPriceMetrics.MAPE - calibratedReport.LowPriceMetrics.MAPE:F2}pp, High={basicReport.HighPriceMetrics.MAPE - calibratedReport.HighPriceMetrics.MAPE:F2}pp");

                    // Feature engineering effectiveness summary
                    Console.WriteLine($"\n🎯 UNIFIED SYSTEM SUMMARY:");
                    Console.WriteLine($"   ✅ Single MarketFeatureEngine handles both basic and enhanced features");
                    Console.WriteLine($"   ✅ Single BayesianStockModel handles both feature types");
                    Console.WriteLine($"   ✅ Automatic feature detection and processing");
                    Console.WriteLine($"   ✅ Hold-out bias correction calibration");
                    Console.WriteLine($"   ✅ Walk-forward validation framework");
                    Console.WriteLine($"   ✅ Integrated temporal/cyclical and technical features");
                    Console.WriteLine($"   📊 Walk-forward average MAPE: {walkForwardResult.AverageCalibratedLowMAPE:F2}%/{walkForwardResult.AverageCalibratedHighMAPE:F2}%");
                    Console.WriteLine($"   📊 Walk-forward directional accuracy: {walkForwardResult.AverageDirectionalAccuracy:F1}%");
                    Console.WriteLine($"   📊 Enhancement value: {basicReport.LowPriceMetrics.MAPE - calibratedReport.LowPriceMetrics.MAPE:F2}pp improvement over basic model");
                }
                else
                {
                    Console.WriteLine($"❌ Insufficient data for training. Need at least 60 samples, got {enhancedFeatures.Count}");
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
    }
}