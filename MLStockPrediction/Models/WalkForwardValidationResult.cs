namespace MLStockPrediction.Models
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

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
            if (this.Folds.Count == 0)
            {
                return;
            }

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
}