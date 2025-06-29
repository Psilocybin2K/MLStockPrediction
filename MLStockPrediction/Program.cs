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
            Console.WriteLine("🚀 Starting Enhanced Bayesian + LightGBM Ensemble Stock Prediction");

            StockDataLoader loader = new StockDataLoader();
            EnhancedFeatureEngine featureEngine = new EnhancedFeatureEngine();
            EnsembleStockModel ensembleModel = new EnsembleStockModel();
            StockModelEvaluator evaluator = new StockModelEvaluator();
            WalkForwardValidator walkForwardValidator = new WalkForwardValidator();

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
                Console.WriteLine($"Feature dimensions: ~85 (9 original + 33 technical + 20 temporal + 25 momentum/volume) - ENSEMBLE ENGINE");

                if (enhancedFeatures.Count > 60)
                {
                    // Walk-Forward Validation with Ensemble
                    Console.WriteLine("\n🔄 Performing Walk-Forward Validation with Ensemble Model...");
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

                    Console.WriteLine($"\n📊 Final Ensemble Training Split: {trainData.Count} train, {testData.Count} test");
                    Console.WriteLine($"📅 Training Period: {trainData.First().Date:yyyy-MM-dd} to {trainData.Last().Date:yyyy-MM-dd}");
                    Console.WriteLine($"📅 Testing Period: {testData.First().Date:yyyy-MM-dd} to {testData.Last().Date:yyyy-MM-dd}");

                    // Train ensemble model (Bayesian + LightGBM)
                    Console.WriteLine("\n🎯 Training Ensemble Model (Bayesian + LightGBM)...");
                    ensembleModel.Train(trainData);
                    Console.WriteLine($"✅ Ensemble model trained on {trainData.Count} samples with ~85 features");

                    // Test both uncalibrated and calibrated ensemble
                    Console.WriteLine("\n🔍 Evaluating Ensemble Performance (Uncalibrated vs Calibrated)...");

                    // Test uncalibrated ensemble
                    ensembleModel.EnableCalibration(false);
                    List<EnsemblePredictionResult> uncalibratedEnsembleResults = new List<EnsemblePredictionResult>();

                    foreach (EnhancedMarketFeatures sample in testData)
                    {
                        EnsemblePredictionResult result = ensembleModel.Predict(sample);
                        uncalibratedEnsembleResults.Add(result);
                        ensembleModel.UpdatePerformance(sample); // Update with actual data
                    }

                    // Test calibrated ensemble
                    ensembleModel.EnableCalibration(true);
                    List<EnsemblePredictionResult> calibratedEnsembleResults = new List<EnsemblePredictionResult>();

                    foreach (EnhancedMarketFeatures sample in testData)
                    {
                        EnsemblePredictionResult result = ensembleModel.Predict(sample);
                        calibratedEnsembleResults.Add(result);
                        ensembleModel.UpdatePerformance(sample); // Update with actual data
                    }

                    // Calculate ensemble performance metrics
                    double uncalMAPELow = CalculateEnsembleMAPE(uncalibratedEnsembleResults, testData, true);
                    double uncalMAPEHigh = CalculateEnsembleMAPE(uncalibratedEnsembleResults, testData, false);
                    double calMAPELow = CalculateEnsembleMAPE(calibratedEnsembleResults, testData, true);
                    double calMAPEHigh = CalculateEnsembleMAPE(calibratedEnsembleResults, testData, false);

                    Console.WriteLine("\n📊 ENSEMBLE MODEL PERFORMANCE:");
                    Console.WriteLine($"   Uncalibrated MAPE: Low={uncalMAPELow:F2}%, High={uncalMAPEHigh:F2}%");
                    Console.WriteLine($"   Calibrated MAPE: Low={calMAPELow:F2}%, High={calMAPEHigh:F2}%");
                    Console.WriteLine($"   Calibration Improvement: Low={uncalMAPELow - calMAPELow:F2}pp, High={uncalMAPEHigh - calMAPEHigh:F2}pp");

                    // Print ensemble statistics
                    ensembleModel.PrintEnsembleStatistics();

                    // Make final prediction on latest data
                    EnhancedMarketFeatures latest = enhancedFeatures.Last();
                    Console.WriteLine($"\n🔮 Making Final Ensemble Prediction for {latest.Date:yyyy-MM-dd}...");

                    // Final prediction with calibration enabled
                    ensembleModel.EnableCalibration(true);
                    EnsemblePredictionResult finalResult = ensembleModel.Predict(latest);

                    Console.WriteLine($"\n=== ENSEMBLE PREDICTION BREAKDOWN for {latest.Date:yyyy-MM-dd} ===");
                    Console.WriteLine($"Actual: Low=${latest.MsftLow:F2}, High=${latest.MsftHigh:F2}");
                    Console.WriteLine($"");
                    Console.WriteLine($"🧠 Bayesian Model: Low=${finalResult.BayesianLow:F2}, High=${finalResult.BayesianHigh:F2}");
                    Console.WriteLine($"🌟 LightGBM Model: Low=${finalResult.LightGbmLow:F2}, High=${finalResult.LightGbmHigh:F2}, Range=${finalResult.LightGbmRange:F2}");
                    Console.WriteLine($"🎯 Final Ensemble: Low=${finalResult.FinalLow:F2}, High=${finalResult.FinalHigh:F2}");
                    Console.WriteLine($"");
                    Console.WriteLine($"📊 Model Weights: Bayesian={finalResult.Weights.BayesianWeight:F3}, LightGBM={finalResult.Weights.LightGbmLowWeight:F3}");
                    Console.WriteLine($"🎭 Market Regime: {finalResult.MarketRegime}");
                    Console.WriteLine($"📈 Bayesian Confidence: {finalResult.BayesianConfidence:F3}");
                    Console.WriteLine($"");
                    Console.WriteLine($"🎯 Ensemble Errors: Low={Math.Abs(latest.MsftLow - finalResult.FinalLow) / latest.MsftLow * 100:F2}%, High={Math.Abs(latest.MsftHigh - finalResult.FinalHigh) / latest.MsftHigh * 100:F2}%");

                    // Display enhanced technical indicators
                    Console.WriteLine($"\n📊 Enhanced Technical Indicators for Latest Prediction:");
                    Console.WriteLine($"   MSFT RSI: {latest.MsftRSI:F2}");
                    Console.WriteLine($"   MSFT MACD: {latest.MsftMACD:F4}");
                    Console.WriteLine($"   MSFT Stochastic: {latest.MsftStochastic:F2}");
                    Console.WriteLine($"   MSFT Volume ROC: {latest.MsftVolumeROC:F4}");
                    Console.WriteLine($"   MSFT VWAP: ${latest.MsftVWAP:F2}");
                    Console.WriteLine($"   MSFT Volatility Ratio: {latest.MsftVolatilityRatio:F2}");
                    Console.WriteLine($"   MSFT True Range: {latest.MsftTrueRange:F4}");

                    Console.WriteLine($"\n📅 Temporal & Regime Features:");
                    Console.WriteLine($"   Day of Week: {latest.Date.DayOfWeek}");
                    Console.WriteLine($"   Options Expiration Week: {latest.IsOptionsExpirationWeek == 1.0}");
                    Console.WriteLine($"   High Volatility Regime: {latest.IsHighVolatilityRegime == 1.0}");
                    Console.WriteLine($"   Strong Trend: Up={latest.IsStrongUptrend == 1.0}, Down={latest.IsStrongDowntrend == 1.0}");
                    Console.WriteLine($"   Quarter Progress: {latest.QuarterProgress:F2}");

                    // Compare with basic Bayesian model for baseline
                    Console.WriteLine($"\n🔄 Comparing with Basic Bayesian Model...");

                    // Create basic features for comparison
                    List<MarketFeatures> basicFeatures = featureEngine.CreateMarketFeatures(allStockData);
                    List<MarketFeatures> basicTrainData = basicFeatures.Take(trainSize).ToList();
                    List<MarketFeatures> basicTestData = basicFeatures.Skip(trainSize).ToList();

                    // Train basic Bayesian model
                    BayesianStockModel basicModel = new BayesianStockModel();
                    basicModel.Train(basicTrainData);
                    basicModel.EnableCalibration(true);

                    // Evaluate basic model
                    StockEvaluationReport basicReport = evaluator.EvaluateModel(basicModel, basicTestData);
                    Console.WriteLine("\n📊 BASIC BAYESIAN MODEL PERFORMANCE (62 features):");
                    evaluator.PrintEvaluationReport(basicReport);

                    // Model comparison summary
                    Console.WriteLine("\n📈 ENSEMBLE vs BASIC MODEL COMPARISON:");
                    Console.WriteLine($"   Ensemble MAPE: Low={calMAPELow:F2}%, High={calMAPEHigh:F2}%");
                    Console.WriteLine($"   Basic Bayesian MAPE: Low={basicReport.LowPriceMetrics.MAPE:F2}%, High={basicReport.HighPriceMetrics.MAPE:F2}%");
                    Console.WriteLine($"   Ensemble Improvement: Low={basicReport.LowPriceMetrics.MAPE - calMAPELow:F2}pp, High={basicReport.HighPriceMetrics.MAPE - calMAPEHigh:F2}pp");

                    // Final system summary
                    Console.WriteLine($"\n🎯 ENSEMBLE SYSTEM SUMMARY:");
                    Console.WriteLine($"   ✅ 2-Layer Ensemble: Bayesian + LightGBM models");
                    Console.WriteLine($"   ✅ Dynamic weight adjustment based on market regime");
                    Console.WriteLine($"   ✅ Enhanced feature engineering: ~85 features");
                    Console.WriteLine($"   ✅ Range validation using auxiliary LightGBM model");
                    Console.WriteLine($"   ✅ Hold-out bias correction calibration");
                    Console.WriteLine($"   ✅ Market regime detection and adaptation");
                    Console.WriteLine($"   ✅ Uncertainty quantification from Bayesian component");
                    Console.WriteLine($"   ✅ Pattern recognition from LightGBM component");
                    Console.WriteLine($"   📊 Walk-forward average MAPE: {walkForwardResult.AverageCalibratedLowMAPE:F2}%/{walkForwardResult.AverageCalibratedHighMAPE:F2}%");
                    Console.WriteLine($"   📊 Walk-forward directional accuracy: {walkForwardResult.AverageDirectionalAccuracy:F1}%");
                    Console.WriteLine($"   📊 Ensemble improvement: {basicReport.LowPriceMetrics.MAPE - calMAPELow:F2}pp over basic model");
                    Console.WriteLine($"   🎭 Current market regime: {finalResult.MarketRegime}");
                    Console.WriteLine($"   ⚖️  Current weights: Bayesian={finalResult.Weights.BayesianWeight:F2}, LightGBM={finalResult.Weights.LightGbmLowWeight:F2}");
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