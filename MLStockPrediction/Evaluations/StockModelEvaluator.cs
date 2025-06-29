namespace MLStockPrediction.Evaluations
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MLStockPrediction.Models;

    public class StockModelEvaluator
    {
        public StockEvaluationReport EvaluateEnhancedModel(
            EnhancedBayesianStockModel model,
            List<EnhancedMarketFeatures> testData)
        {
            Console.WriteLine($"🔍 Evaluating enhanced model on {testData.Count} test samples...");

            StockEvaluationReport report = new StockEvaluationReport();
            List<StockPredictionResult> predictions = new List<StockPredictionResult>();

            foreach (EnhancedMarketFeatures marketData in testData)
            {
                Console.WriteLine($"\n📅 Enhanced Predicting for {marketData.Date:yyyy-MM-dd}");
                Console.WriteLine($"Actual: Low=${marketData.MsftLow:F2}, High=${marketData.MsftHigh:F2}");

                (double predictedLow, double predictedHigh) = model.Predict(marketData);

                predictions.Add(new StockPredictionResult
                {
                    Date = marketData.Date,
                    ActualLow = marketData.MsftLow,
                    ActualHigh = marketData.MsftHigh,
                    PredictedLow = predictedLow,
                    PredictedHigh = predictedHigh,
                    LowError = Math.Abs(marketData.MsftLow - predictedLow),
                    HighError = Math.Abs(marketData.MsftHigh - predictedHigh),
                    LowPercentError = Math.Abs((marketData.MsftLow - predictedLow) / marketData.MsftLow) * 100,
                    HighPercentError = Math.Abs((marketData.MsftHigh - predictedHigh) / marketData.MsftHigh) * 100
                });

                Console.WriteLine($"Predicted: Low=${predictedLow:F2}, High=${predictedHigh:F2}");
                Console.WriteLine($"Errors: Low={predictions.Last().LowPercentError:F2}%, High={predictions.Last().HighPercentError:F2}%");
            }

            report.SampleSize = predictions.Count;
            report.LowPriceMetrics = this.CalculatePriceMetrics(predictions, true);
            report.HighPriceMetrics = this.CalculatePriceMetrics(predictions, false);
            report.DirectionalAccuracy = this.CalculateDirectionalAccuracy(predictions);
            report.ErrorDistribution = this.AnalyzeErrorDistribution(predictions);

            return report;
        }

        public StockEvaluationReport EvaluateModel(
            BayesianStockModel model,
            List<MarketFeatures> testData)
        {
            Console.WriteLine($"🔍 Evaluating model on {testData.Count} test samples...");

            StockEvaluationReport report = new StockEvaluationReport();
            List<StockPredictionResult> predictions = new List<StockPredictionResult>();

            foreach (MarketFeatures marketData in testData)
            {
                Console.WriteLine($"\n📅 Predicting for {marketData.Date:yyyy-MM-dd}");
                Console.WriteLine($"Actual: Low=${marketData.MsftLow:F2}, High=${marketData.MsftHigh:F2}");

                (double predictedLow, double predictedHigh) = model.Predict(marketData);

                predictions.Add(new StockPredictionResult
                {
                    Date = marketData.Date,
                    ActualLow = marketData.MsftLow,
                    ActualHigh = marketData.MsftHigh,
                    PredictedLow = predictedLow,
                    PredictedHigh = predictedHigh,
                    LowError = Math.Abs(marketData.MsftLow - predictedLow),
                    HighError = Math.Abs(marketData.MsftHigh - predictedHigh),
                    LowPercentError = Math.Abs((marketData.MsftLow - predictedLow) / marketData.MsftLow) * 100,
                    HighPercentError = Math.Abs((marketData.MsftHigh - predictedHigh) / marketData.MsftHigh) * 100
                });

                Console.WriteLine($"Predicted: Low=${predictedLow:F2}, High=${predictedHigh:F2}");
                Console.WriteLine($"Errors: Low={predictions.Last().LowPercentError:F2}%, High={predictions.Last().HighPercentError:F2}%");
            }

            report.SampleSize = predictions.Count;
            report.LowPriceMetrics = this.CalculatePriceMetrics(predictions, true);
            report.HighPriceMetrics = this.CalculatePriceMetrics(predictions, false);
            report.DirectionalAccuracy = this.CalculateDirectionalAccuracy(predictions);
            report.ErrorDistribution = this.AnalyzeErrorDistribution(predictions);

            return report;
        }

        private PriceMetrics CalculatePriceMetrics(List<StockPredictionResult> predictions, bool isLow)
        {
            List<double> errors = isLow
                ? predictions.Select(p => p.LowError).ToList()
                : predictions.Select(p => p.HighError).ToList();

            List<double> percentErrors = isLow
                ? predictions.Select(p => p.LowPercentError).ToList()
                : predictions.Select(p => p.HighPercentError).ToList();

            // Calculate accuracy across 0-100% in steps of 10%
            Dictionary<string, double> accuracyBreakdown = new Dictionary<string, double>();
            for (int threshold = 10; threshold <= 100; threshold += 10)
            {
                double accuracy = predictions.Count(p =>
                    (isLow ? p.LowPercentError : p.HighPercentError) <= threshold) / (double)predictions.Count * 100;
                accuracyBreakdown[$"≤{threshold}%"] = accuracy;
            }

            return new PriceMetrics
            {
                MAE = errors.Average(),
                RMSE = Math.Sqrt(errors.Select(e => e * e).Average()),
                MAPE = percentErrors.Average(),
                MedianError = errors.OrderBy(x => x).Skip(errors.Count / 2).First(),
                MaxError = errors.Max(),
                AccuracyWithin1Percent = predictions.Count(p =>
                    (isLow ? p.LowPercentError : p.HighPercentError) <= 1.0) / (double)predictions.Count * 100,
                AccuracyWithin5Percent = predictions.Count(p =>
                    (isLow ? p.LowPercentError : p.HighPercentError) <= 5.0) / (double)predictions.Count * 100,
                AccuracyBreakdown = accuracyBreakdown
            };
        }

        private DirectionalAccuracy CalculateDirectionalAccuracy(List<StockPredictionResult> predictions)
        {
            int correctLowDirection = 0;
            int correctHighDirection = 0;
            int correctRangeDirection = 0;

            for (int i = 1; i < predictions.Count; i++)
            {
                StockPredictionResult current = predictions[i];
                StockPredictionResult previous = predictions[i - 1];

                // Low price direction
                int actualLowDirection = Math.Sign(current.ActualLow - previous.ActualLow);
                int predictedLowDirection = Math.Sign(current.PredictedLow - previous.PredictedLow);
                if (actualLowDirection == predictedLowDirection)
                {
                    correctLowDirection++;
                }

                // High price direction
                int actualHighDirection = Math.Sign(current.ActualHigh - previous.ActualHigh);
                int predictedHighDirection = Math.Sign(current.PredictedHigh - previous.PredictedHigh);
                if (actualHighDirection == predictedHighDirection)
                {
                    correctHighDirection++;
                }

                // Range direction
                double actualRange = current.ActualHigh - current.ActualLow;
                double previousActualRange = previous.ActualHigh - previous.ActualLow;
                double predictedRange = current.PredictedHigh - current.PredictedLow;
                double previousPredictedRange = previous.PredictedHigh - previous.PredictedLow;

                int actualRangeDirection = Math.Sign(actualRange - previousActualRange);
                int predictedRangeDirection = Math.Sign(predictedRange - previousPredictedRange);
                if (actualRangeDirection == predictedRangeDirection)
                {
                    correctRangeDirection++;
                }
            }

            int totalComparisons = predictions.Count - 1;
            return new DirectionalAccuracy
            {
                LowDirectionalAccuracy = correctLowDirection / (double)totalComparisons * 100,
                HighDirectionalAccuracy = correctHighDirection / (double)totalComparisons * 100,
                RangeDirectionalAccuracy = correctRangeDirection / (double)totalComparisons * 100
            };
        }

        private StockErrorDistribution AnalyzeErrorDistribution(List<StockPredictionResult> predictions)
        {
            List<double> lowErrors = predictions.Select(p => p.LowPercentError).OrderBy(x => x).ToList();
            List<double> highErrors = predictions.Select(p => p.HighPercentError).OrderBy(x => x).ToList();

            return new StockErrorDistribution
            {
                LowErrorPercentiles = new Dictionary<string, double>
                {
                    ["P25"] = this.GetPercentile(lowErrors, 0.25),
                    ["P50"] = this.GetPercentile(lowErrors, 0.5),
                    ["P75"] = this.GetPercentile(lowErrors, 0.75),
                    ["P90"] = this.GetPercentile(lowErrors, 0.9),
                    ["P95"] = this.GetPercentile(lowErrors, 0.95)
                },
                HighErrorPercentiles = new Dictionary<string, double>
                {
                    ["P25"] = this.GetPercentile(highErrors, 0.25),
                    ["P50"] = this.GetPercentile(highErrors, 0.5),
                    ["P75"] = this.GetPercentile(highErrors, 0.75),
                    ["P90"] = this.GetPercentile(highErrors, 0.9),
                    ["P95"] = this.GetPercentile(highErrors, 0.95)
                }
            };
        }

        private double GetPercentile(List<double> sortedValues, double percentile)
        {
            int index = (int)(percentile * (sortedValues.Count - 1));
            return sortedValues[index];
        }

        public void PrintEvaluationReport(StockEvaluationReport report)
        {
            Console.WriteLine("\n" + "=".PadRight(60, '='));
            Console.WriteLine("📊 ENHANCED STOCK PREDICTION MODEL EVALUATION");
            Console.WriteLine("=".PadRight(60, '='));

            Console.WriteLine($"\n🎯 SAMPLE SIZE: {report.SampleSize:N0}");

            Console.WriteLine($"\n📈 LOW PRICE PREDICTIONS");
            Console.WriteLine($"   MAE: ${report.LowPriceMetrics.MAE:F3}");
            Console.WriteLine($"   RMSE: ${report.LowPriceMetrics.RMSE:F3}");
            Console.WriteLine($"   MAPE: {report.LowPriceMetrics.MAPE:F2}%");
            Console.WriteLine($"   Accuracy ≤1%: {report.LowPriceMetrics.AccuracyWithin1Percent:F1}%");
            Console.WriteLine($"   Accuracy ≤5%: {report.LowPriceMetrics.AccuracyWithin5Percent:F1}%");

            // Print accuracy breakdown 0-100% in steps of 10%
            Console.WriteLine("   📊 Accuracy Breakdown:");
            if (report.LowPriceMetrics.AccuracyBreakdown != null)
            {
                foreach (KeyValuePair<string, double> kvp in report.LowPriceMetrics.AccuracyBreakdown.OrderBy(x => x.Key))
                {
                    Console.WriteLine($"      {kvp.Key}: {kvp.Value:F1}%");
                }
            }

            Console.WriteLine($"\n📈 HIGH PRICE PREDICTIONS");
            Console.WriteLine($"   MAE: ${report.HighPriceMetrics.MAE:F3}");
            Console.WriteLine($"   RMSE: ${report.HighPriceMetrics.RMSE:F3}");
            Console.WriteLine($"   MAPE: {report.HighPriceMetrics.MAPE:F2}%");
            Console.WriteLine($"   Accuracy ≤1%: {report.HighPriceMetrics.AccuracyWithin1Percent:F1}%");
            Console.WriteLine($"   Accuracy ≤5%: {report.HighPriceMetrics.AccuracyWithin5Percent:F1}%");

            // Print accuracy breakdown 0-100% in steps of 10%
            Console.WriteLine("   📊 Accuracy Breakdown:");
            if (report.HighPriceMetrics.AccuracyBreakdown != null)
            {
                foreach (KeyValuePair<string, double> kvp in report.HighPriceMetrics.AccuracyBreakdown.OrderBy(x => x.Key))
                {
                    Console.WriteLine($"      {kvp.Key}: {kvp.Value:F1}%");
                }
            }

            Console.WriteLine($"\n🎯 DIRECTIONAL ACCURACY");
            Console.WriteLine($"   Low Price Direction: {report.DirectionalAccuracy.LowDirectionalAccuracy:F1}%");
            Console.WriteLine($"   High Price Direction: {report.DirectionalAccuracy.HighDirectionalAccuracy:F1}%");
            Console.WriteLine($"   Range Direction: {report.DirectionalAccuracy.RangeDirectionalAccuracy:F1}%");

            Console.WriteLine($"\n📊 ERROR DISTRIBUTION");
            Console.WriteLine($"   Low Price P50: {report.ErrorDistribution.LowErrorPercentiles["P50"]:F2}%");
            Console.WriteLine($"   Low Price P95: {report.ErrorDistribution.LowErrorPercentiles["P95"]:F2}%");
            Console.WriteLine($"   High Price P50: {report.ErrorDistribution.HighErrorPercentiles["P50"]:F2}%");
            Console.WriteLine($"   High Price P95: {report.ErrorDistribution.HighErrorPercentiles["P95"]:F2}%");
        }
    }
}