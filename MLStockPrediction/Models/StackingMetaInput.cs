namespace MLStockPrediction.Models
{
    public class StackingMetaInput
    {
        // Base model predictions
        public float BayesianLow { get; set; }
        public float BayesianHigh { get; set; }
        public float LightGbmLow { get; set; }
        public float LightGbmHigh { get; set; }

        // New features for additional context
        public float MsftVolatility { get; set; }
        public float MsftRSI { get; set; }

        // The actual target values for training the meta-model
        public float ActualLow { get; set; }
        public float ActualHigh { get; set; }
    }
}
