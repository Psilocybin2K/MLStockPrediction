namespace MLStockPrediction.Models
{
    using System;

    public class StockPredictionResult
    {
        public DateTime Date { get; set; }
        public double ActualLow { get; set; }
        public double ActualHigh { get; set; }
        public double PredictedLow { get; set; }
        public double PredictedHigh { get; set; }
        public double LowError { get; set; }
        public double HighError { get; set; }
        public double LowPercentError { get; set; }
        public double HighPercentError { get; set; }
    }
}