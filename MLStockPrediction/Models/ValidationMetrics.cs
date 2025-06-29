namespace MLStockPrediction.Models
{
    public class ValidationMetrics
    {
        public double LowMAPE { get; set; }
        public double HighMAPE { get; set; }
        public double LowMAE { get; set; }
        public double HighMAE { get; set; }
        public double DirectionalAccuracy { get; set; }
        public int SampleCount { get; set; }
    }
}