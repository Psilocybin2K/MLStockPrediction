namespace MLStockPrediction.Models
{
    public class StockEvaluationReport
    {
        public int SampleSize { get; set; }
        public PriceMetrics LowPriceMetrics { get; set; }
        public PriceMetrics HighPriceMetrics { get; set; }
        public DirectionalAccuracy DirectionalAccuracy { get; set; }
        public StockErrorDistribution ErrorDistribution { get; set; }
    }
}
