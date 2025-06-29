namespace MLStockPrediction.Models
{
    using System;

    public class MarketFeatures
    {
        public DateTime Date { get; set; }

        // DOW features
        public double DowReturn { get; set; }
        public double DowVolatility { get; set; }
        public double DowVolume { get; set; }

        // QQQ features
        public double QqqReturn { get; set; }
        public double QqqVolatility { get; set; }
        public double QqqVolume { get; set; }

        // MSFT features
        public double MsftReturn { get; set; }
        public double MsftVolatility { get; set; }
        public double MsftVolume { get; set; }

        // Correlation features
        public double DowMsftCorrelation { get; set; }
        public double QqqMsftCorrelation { get; set; }

        // Target variables
        public double MsftLow { get; set; }
        public double MsftHigh { get; set; }
    }
}
