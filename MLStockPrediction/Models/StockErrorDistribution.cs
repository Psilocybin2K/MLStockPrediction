namespace MLStockPrediction.Models
{
    public class StockErrorDistribution
    {
        public Dictionary<string, double> LowErrorPercentiles { get; set; }
        public Dictionary<string, double> HighErrorPercentiles { get; set; }
    }
}