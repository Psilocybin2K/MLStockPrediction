namespace MLStockPrediction.Evaluations
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MLStockPrediction.Models;

    public class WalkForwardValidator
    {
        public WalkForwardValidationResult ValidateModel(
            List<EnhancedMarketFeatures> allData,
            int initialTrainingSize = 50,
            int validationWindow = 10,
            int stepSize = 5)
        {
            Console.WriteLine("🔄 Starting Walk-Forward Validation with Hold-Out Calibration...");
            Console.WriteLine($"   Initial training size: {initialTrainingSize}");
            Console.WriteLine($"   Validation window: {validationWindow}");
            Console.WriteLine($"   Step size: {stepSize}");

            WalkForwardValidationResult result = new WalkForwardValidationResult();
            List<WalkForwardFold> foldResults = new List<WalkForwardFold>();

            int totalSteps = (allData.Count - initialTrainingSize - validationWindow) / stepSize + 1;
            Console.WriteLine($"   Total validation steps: {totalSteps}");

            for (int step = 0; step < totalSteps; step++)
            {
                int trainStart = 0;
                int trainEnd = initialTrainingSize + step * stepSize;
                int validStart = trainEnd;
                int validEnd = Math.Min(validStart + validationWindow, allData.Count);

                if (validEnd <= validStart)
                {
                    break;
                }

                Console.WriteLine($"\n📊 Walk-Forward Step {step + 1}/{totalSteps}");
                Console.WriteLine($"   Training: [{trainStart}-{trainEnd}] ({trainEnd - trainStart} samples)");
                Console.WriteLine($"   Validation: [{validStart}-{validEnd}] ({validEnd - validStart} samples)");
                Console.WriteLine($"   Date range: {allData[trainStart].Date:yyyy-MM-dd} to {allData[validEnd - 1].Date:yyyy-MM-dd}");

                // Extract training and validation data
                List<EnhancedMarketFeatures> trainData = allData.GetRange(trainStart, trainEnd - trainStart);
                List<EnhancedMarketFeatures> validData = allData.GetRange(validStart, validEnd - validStart);

                // FIXED: Train model (now uses hold-out calibration internally)
                EnhancedBayesianStockModel model = new EnhancedBayesianStockModel();
                model.Train(trainData);

                // Test without calibration first
                model.EnableCalibration(false);
                ValidationMetrics uncalibratedResults = this.EvaluateFold(model, validData, "Uncalibrated");

                // Test with hold-out calibration
                model.EnableCalibration(true);
                ValidationMetrics calibratedResults = this.EvaluateFold(model, validData, "Hold-Out Calibrated");

                WalkForwardFold fold = new WalkForwardFold
                {
                    StepNumber = step + 1,
                    TrainingStartDate = allData[trainStart].Date,
                    TrainingEndDate = allData[trainEnd - 1].Date,
                    ValidationStartDate = allData[validStart].Date,
                    ValidationEndDate = allData[validEnd - 1].Date,
                    TrainingSamples = trainEnd - trainStart,
                    ValidationSamples = validEnd - validStart,
                    UncalibratedResults = uncalibratedResults,
                    CalibratedResults = calibratedResults
                };

                foldResults.Add(fold);

                Console.WriteLine($"   Uncalibrated MAPE: Low={uncalibratedResults.LowMAPE:F2}%, High={uncalibratedResults.HighMAPE:F2}%");
                Console.WriteLine($"   Hold-Out Calibrated MAPE: Low={calibratedResults.LowMAPE:F2}%, High={calibratedResults.HighMAPE:F2}%");
                Console.WriteLine($"   Hold-Out Improvement: Low={uncalibratedResults.LowMAPE - calibratedResults.LowMAPE:F2}pp, High={uncalibratedResults.HighMAPE - calibratedResults.HighMAPE:F2}pp");
            }

            result.Folds = foldResults;
            result.CalculateAverageMetrics();

            Console.WriteLine("\n📈 Walk-Forward Validation Summary (Hold-Out Calibration):");
            Console.WriteLine($"   Average Uncalibrated MAPE: Low={result.AverageUncalibratedLowMAPE:F2}%, High={result.AverageUncalibratedHighMAPE:F2}%");
            Console.WriteLine($"   Average Hold-Out Calibrated MAPE: Low={result.AverageCalibratedLowMAPE:F2}%, High={result.AverageCalibratedHighMAPE:F2}%");
            Console.WriteLine($"   Average Hold-Out Improvement: Low={result.AverageUncalibratedLowMAPE - result.AverageCalibratedLowMAPE:F2}pp, High={result.AverageUncalibratedHighMAPE - result.AverageCalibratedHighMAPE:F2}pp");
            Console.WriteLine($"   Average Directional Accuracy: {result.AverageDirectionalAccuracy:F1}%");

            return result;
        }

        private ValidationMetrics EvaluateFold(EnhancedBayesianStockModel model, List<EnhancedMarketFeatures> validationData, string description)
        {
            List<StockPredictionResult> predictions = new List<StockPredictionResult>();

            foreach (EnhancedMarketFeatures sample in validationData)
            {
                (double predLow, double predHigh) = model.Predict(sample);

                predictions.Add(new StockPredictionResult
                {
                    Date = sample.Date,
                    ActualLow = sample.MsftLow,
                    ActualHigh = sample.MsftHigh,
                    PredictedLow = predLow,
                    PredictedHigh = predHigh,
                    LowError = Math.Abs(sample.MsftLow - predLow),
                    HighError = Math.Abs(sample.MsftHigh - predHigh),
                    LowPercentError = Math.Abs((sample.MsftLow - predLow) / sample.MsftLow) * 100,
                    HighPercentError = Math.Abs((sample.MsftHigh - predHigh) / sample.MsftHigh) * 100
                });
            }

            // Calculate directional accuracy
            int correctDirections = 0;
            for (int i = 1; i < predictions.Count; i++)
            {
                StockPredictionResult current = predictions[i];
                StockPredictionResult previous = predictions[i - 1];

                int actualLowDirection = Math.Sign(current.ActualLow - previous.ActualLow);
                int predictedLowDirection = Math.Sign(current.PredictedLow - previous.PredictedLow);

                int actualHighDirection = Math.Sign(current.ActualHigh - previous.ActualHigh);
                int predictedHighDirection = Math.Sign(current.PredictedHigh - previous.PredictedHigh);

                if (actualLowDirection == predictedLowDirection && actualHighDirection == predictedHighDirection)
                {
                    correctDirections++;
                }
            }

            double directionalAccuracy = predictions.Count > 1 ? correctDirections / (double)(predictions.Count - 1) * 100 : 0;

            return new ValidationMetrics
            {
                LowMAPE = predictions.Average(p => p.LowPercentError),
                HighMAPE = predictions.Average(p => p.HighPercentError),
                LowMAE = predictions.Average(p => p.LowError),
                HighMAE = predictions.Average(p => p.HighError),
                DirectionalAccuracy = directionalAccuracy,
                SampleCount = predictions.Count
            };
        }
    }
}