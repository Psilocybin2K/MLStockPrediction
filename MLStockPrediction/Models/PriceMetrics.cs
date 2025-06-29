namespace MLStockPrediction.Models
{
    public class PriceMetrics
    {
        public double MAE { get; set; }
        public double RMSE { get; set; }
        public double MAPE { get; set; }
        public double MedianError { get; set; }
        public double MaxError { get; set; }
        public double AccuracyWithin1Percent { get; set; }
        public double AccuracyWithin5Percent { get; set; }
        public Dictionary<string, double> AccuracyBreakdown { get; set; } = new Dictionary<string, double>();
    }
}
