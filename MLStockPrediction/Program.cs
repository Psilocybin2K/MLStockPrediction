namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;

    using MLStockPrediction.Models;

    public class WalkForwardValidationResult
    {
        public List<WalkForwardFold> Folds { get; set; } = new List<WalkForwardFold>();

        // Average metrics
        public double AverageUncalibratedLowMAPE { get; set; }
        public double AverageUncalibratedHighMAPE { get; set; }
        public double AverageCalibratedLowMAPE { get; set; }
        public double AverageCalibratedHighMAPE { get; set; }
        public double AverageDirectionalAccuracy { get; set; }

        public void CalculateAverageMetrics()
        {
            if (this.Folds.Count == 0) return;

            this.AverageUncalibratedLowMAPE = this.Folds.Average(f => f.UncalibratedResults.LowMAPE);
            this.AverageUncalibratedHighMAPE = this.Folds.Average(f => f.UncalibratedResults.HighMAPE);
            this.AverageCalibratedLowMAPE = this.Folds.Average(f => f.CalibratedResults.LowMAPE);
            this.AverageCalibratedHighMAPE = this.Folds.Average(f => f.CalibratedResults.HighMAPE);
            this.AverageDirectionalAccuracy = this.Folds.Average(f => f.CalibratedResults.DirectionalAccuracy);
        }

        public void PrintDetailedReport()
        {
            Console.WriteLine("\n" + "=".PadRight(80, '='));
            Console.WriteLine("📊 WALK-FORWARD VALIDATION DETAILED REPORT (HOLD-OUT CALIBRATION)");
            Console.WriteLine("=".PadRight(80, '='));

            Console.WriteLine($"\n🎯 OVERALL PERFORMANCE SUMMARY");
            Console.WriteLine($"   Total validation folds: {this.Folds.Count}");
            Console.WriteLine($"   Uncalibrated Average MAPE: Low={this.AverageUncalibratedLowMAPE:F2}%, High={this.AverageUncalibratedHighMAPE:F2}%");
            Console.WriteLine($"   Hold-Out Calibrated Average MAPE: Low={this.AverageCalibratedLowMAPE:F2}%, High={this.AverageCalibratedHighMAPE:F2}%");
            Console.WriteLine($"   Hold-Out Calibration Improvement: Low={this.AverageUncalibratedLowMAPE - this.AverageCalibratedLowMAPE:F2}pp, High={this.AverageUncalibratedHighMAPE - this.AverageCalibratedHighMAPE:F2}pp");
            Console.WriteLine($"   Average Directional Accuracy: {this.AverageDirectionalAccuracy:F1}%");

            Console.WriteLine($"\n📈 FOLD-BY-FOLD BREAKDOWN");
            Console.WriteLine("Step | Date Range           | Samples | Uncalib MAPE | Hold-Out MAPE | Improvement | Dir Acc");
            Console.WriteLine("-".PadRight(80, '-'));

            foreach (WalkForwardFold fold in this.Folds)
            {
                string dateRange = $"{fold.ValidationStartDate:MM/dd} - {fold.ValidationEndDate:MM/dd}";
                string uncalibMAPE = $"{fold.UncalibratedResults.LowMAPE:F1}%/{fold.UncalibratedResults.HighMAPE:F1}%";
                string calibMAPE = $"{fold.CalibratedResults.LowMAPE:F1}%/{fold.CalibratedResults.HighMAPE:F1}%";
                string improvement = $"{fold.UncalibratedResults.LowMAPE - fold.CalibratedResults.LowMAPE:+F1;-F1}pp/{fold.UncalibratedResults.HighMAPE - fold.CalibratedResults.HighMAPE:+F1;-F1}pp";

                Console.WriteLine($"{fold.StepNumber,4} | {dateRange,-19} | {fold.ValidationSamples,7} | {uncalibMAPE,-12} | {calibMAPE,-13} | {improvement,-11} | {fold.CalibratedResults.DirectionalAccuracy,6:F1}%");
            }

            Console.WriteLine("\n📊 PERFORMANCE STABILITY ANALYSIS");
            if (this.Folds.Count > 1)
            {
                List<double> calibratedLowMAPEs = this.Folds.Select(f => f.CalibratedResults.LowMAPE).ToList();
                List<double> calibratedHighMAPEs = this.Folds.Select(f => f.CalibratedResults.HighMAPE).ToList();
                List<double> improvements = this.Folds.Select(f => f.UncalibratedResults.LowMAPE - f.CalibratedResults.LowMAPE).ToList();

                double lowStdDev = Math.Sqrt(calibratedLowMAPEs.Select(x => Math.Pow(x - this.AverageCalibratedLowMAPE, 2)).Average());
                double highStdDev = Math.Sqrt(calibratedHighMAPEs.Select(x => Math.Pow(x - this.AverageCalibratedHighMAPE, 2)).Average());
                double improvementStdDev = Math.Sqrt(improvements.Select(x => Math.Pow(x - improvements.Average(), 2)).Average());

                Console.WriteLine($"   Hold-Out Calibrated Low MAPE Std Dev: {lowStdDev:F2}% (CV: {lowStdDev / this.AverageCalibratedLowMAPE:F2})");
                Console.WriteLine($"   Hold-Out Calibrated High MAPE Std Dev: {highStdDev:F2}% (CV: {highStdDev / this.AverageCalibratedHighMAPE:F2})");
                Console.WriteLine($"   Hold-Out Improvement Consistency: {improvementStdDev:F2}pp std dev");
                Console.WriteLine($"   Best Hold-Out Low MAPE: {calibratedLowMAPEs.Min():F2}% (Fold {this.Folds[calibratedLowMAPEs.IndexOf(calibratedLowMAPEs.Min())].StepNumber})");
                Console.WriteLine($"   Worst Hold-Out Low MAPE: {calibratedLowMAPEs.Max():F2}% (Fold {this.Folds[calibratedLowMAPEs.IndexOf(calibratedLowMAPEs.Max())].StepNumber})");

                // Check if hold-out calibration is consistently helpful
                int positiveImprovements = improvements.Count(x => x > 0);
                Console.WriteLine($"   Hold-Out Calibration Success Rate: {positiveImprovements}/{improvements.Count} ({100.0 * positiveImprovements / improvements.Count:F1}%)");
            }
        }
    }

    public class ValidationMetrics
    {
        public double LowMAPE { get; set; }
        public double HighMAPE { get; set; }
        public double LowMAE { get; set; }
        public double HighMAE { get; set; }
        public double DirectionalAccuracy { get; set; }
        public int SampleCount { get; set; }
    }

    public class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("🚀 Starting Enhanced Bayesian Stock Prediction with Fixed Features");

            StockDataLoader loader = new StockDataLoader();
            EnhancedMarketFeatureEngine featureEngine = new EnhancedMarketFeatureEngine();
            EnhancedBayesianStockModel enhancedModel = new EnhancedBayesianStockModel();
            StockModelEvaluator evaluator = new StockModelEvaluator();
            WalkForwardValidator walkForwardValidator = new WalkForwardValidator();

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
                Console.WriteLine($"Feature dimensions: 62 (9 original + 33 technical + 20 temporal) - FIXED");

                if (enhancedFeatures.Count > 60)
                {
                    // NEW: Walk-Forward Validation
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

                    // Train final enhanced Bayesian model
                    Console.WriteLine("\n🎯 Training Final Enhanced Model with Fixed Features...");
                    enhancedModel.Train(trainData);
                    Console.WriteLine($"✅ Enhanced Bayesian model trained on {trainData.Count} samples with exactly 62 features");

                    // Test both uncalibrated and calibrated models
                    Console.WriteLine("\n🔍 Evaluating Model Performance (Uncalibrated vs Calibrated)...");

                    // Test uncalibrated first
                    enhancedModel.EnableCalibration(false);
                    StockEvaluationReport uncalibratedReport = evaluator.EvaluateEnhancedModel(enhancedModel, testData);
                    Console.WriteLine("\n📊 UNCALIBRATED MODEL PERFORMANCE:");
                    evaluator.PrintEvaluationReport(uncalibratedReport);

                    // Test calibrated
                    enhancedModel.EnableCalibration(true);
                    StockEvaluationReport calibratedReport = evaluator.EvaluateEnhancedModel(enhancedModel, testData);
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
                    enhancedModel.EnableCalibration(false);
                    (double uncalLow, double uncalHigh) = enhancedModel.Predict(latest);

                    // Calibrated prediction
                    enhancedModel.EnableCalibration(true);
                    (double calLow, double calHigh) = enhancedModel.Predict(latest);

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
                    Console.WriteLine($"   MSFT Volatility (20-day): {latest.MsftVolatility20:F4}");
                    Console.WriteLine($"   MSFT ATR: ${latest.MsftATR:F2}");
                    Console.WriteLine($"   MSFT Bollinger Position: {latest.MsftBBPosition:F2} (0=center, ±1=bands)");

                    Console.WriteLine($"\n📅 Temporal Features for Latest Prediction:");
                    Console.WriteLine($"   Day of Week: {latest.Date.DayOfWeek}");
                    Console.WriteLine($"   Options Expiration Week: {latest.IsOptionsExpirationWeek == 1.0}");
                    Console.WriteLine($"   Quarter Progress: {latest.QuarterProgress:F2}");
                    Console.WriteLine($"   Year Progress: {latest.YearProgress:F2}");
                    Console.WriteLine($"   Days to Holiday: {latest.DaysToMarketHoliday}");
                    Console.WriteLine($"   Earnings Season: {latest.IsEarningsSeason == 1.0}");

                    // Feature engineering effectiveness summary
                    Console.WriteLine($"\n🎯 MODEL ENHANCEMENT SUMMARY:");
                    Console.WriteLine($"   ✅ Fixed feature vector size: 62 features exactly");
                    Console.WriteLine($"   ✅ Added bias correction calibration");
                    Console.WriteLine($"   ✅ Implemented walk-forward validation");
                    Console.WriteLine($"   ✅ Integrated 20 temporal/cyclical features");
                    Console.WriteLine($"   ✅ Enhanced with 33 technical indicators");
                    Console.WriteLine($"   📊 Walk-forward average MAPE: {walkForwardResult.AverageCalibratedLowMAPE:F2}%/{walkForwardResult.AverageCalibratedHighMAPE:F2}%");
                    Console.WriteLine($"   📊 Walk-forward directional accuracy: {walkForwardResult.AverageDirectionalAccuracy:F1}%");
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