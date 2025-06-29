namespace MLStockPrediction.Models
{
    using System;

    public class StockFeatures
    {
        public DateTime Date { get; set; }

        // Price-based features
        public double PriceRange { get; set; }
        public double OpenCloseRatio { get; set; }
        public double HighLowRatio { get; set; }
        public double VolumeNormalized { get; set; }

        // Technical indicators
        public double PreviousDayReturn { get; set; }
        public double Volatility { get; set; }
        public double MovingAverage { get; set; }

        // Target variables
        public double Low { get; set; }
        public double High { get; set; }
    }
}
