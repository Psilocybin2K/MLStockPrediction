namespace MLStockPrediction.Models
{
    using System;

    public class WalkForwardFold
    {
        public int StepNumber { get; set; }
        public DateTime TrainingStartDate { get; set; }
        public DateTime TrainingEndDate { get; set; }
        public DateTime ValidationStartDate { get; set; }
        public DateTime ValidationEndDate { get; set; }
        public int TrainingSamples { get; set; }
        public int ValidationSamples { get; set; }
        public ValidationMetrics UncalibratedResults { get; set; }
        public ValidationMetrics CalibratedResults { get; set; }
    }
}