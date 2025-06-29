namespace MLStockPrediction.Models
{
    using Microsoft.ML.Data;

    public class LightGbmInput
    {
        // Original features (9)
        public float DowReturn { get; set; }
        public float DowVolatility { get; set; }
        public float DowVolume { get; set; }
        public float QqqReturn { get; set; }
        public float QqqVolatility { get; set; }
        public float QqqVolume { get; set; }
        public float MsftReturn { get; set; }
        public float MsftVolatility { get; set; }
        public float MsftVolume { get; set; }

        // Technical features - Moving Averages (9)
        public float DowSMA5 { get; set; }
        public float DowSMA10 { get; set; }
        public float DowSMA20 { get; set; }
        public float QqqSMA5 { get; set; }
        public float QqqSMA10 { get; set; }
        public float QqqSMA20 { get; set; }
        public float MsftSMA5 { get; set; }
        public float MsftSMA10 { get; set; }
        public float MsftSMA20 { get; set; }

        // EMA Ratios (9)
        public float DowEMAR5 { get; set; }
        public float DowEMAR10 { get; set; }
        public float DowEMAR20 { get; set; }
        public float QqqEMAR5 { get; set; }
        public float QqqEMAR10 { get; set; }
        public float QqqEMAR20 { get; set; }
        public float MsftEMAR5 { get; set; }
        public float MsftEMAR10 { get; set; }
        public float MsftEMAR20 { get; set; }

        // Price Position (3)
        public float DowPricePosition { get; set; }
        public float QqqPricePosition { get; set; }
        public float MsftPricePosition { get; set; }

        // Rate of Change (6)
        public float DowROC5 { get; set; }
        public float DowROC10 { get; set; }
        public float QqqROC5 { get; set; }
        public float QqqROC10 { get; set; }
        public float MsftROC5 { get; set; }
        public float MsftROC10 { get; set; }

        // Rolling Volatility (6)
        public float DowVolatility5 { get; set; }
        public float DowVolatility10 { get; set; }
        public float QqqVolatility5 { get; set; }
        public float QqqVolatility10 { get; set; }
        public float MsftVolatility5 { get; set; }
        public float MsftVolatility10 { get; set; }

        // Temporal features (10 selected key ones)
        public float IsMondayEffect { get; set; }
        public float IsTuesdayEffect { get; set; }
        public float IsWednesdayEffect { get; set; }
        public float IsThursdayEffect { get; set; }
        public float IsFridayEffect { get; set; }
        public float IsOptionsExpirationWeek { get; set; }
        public float IsQuarterStart { get; set; }
        public float IsQuarterEnd { get; set; }
        public float QuarterProgress { get; set; }
        public float YearProgress { get; set; }

        // Target variables (for training)
        public float LowPrice { get; set; }
        public float HighPrice { get; set; }
        public float PriceRange { get; set; }
    }

    public class LightGbmLowOutput
    {
        [ColumnName("Score")]
        public float PredictedLowPrice { get; set; }
    }

    public class LightGbmHighOutput
    {
        [ColumnName("Score")]
        public float PredictedHighPrice { get; set; }
    }

    public class LightGbmRangeOutput
    {
        [ColumnName("Score")]
        public float PredictedRange { get; set; }
    }

    public class EnsembleWeights
    {
        public double BayesianWeight { get; set; } = 0.5;
        public double LightGbmLowWeight { get; set; } = 0.5;
        public double LightGbmHighWeight { get; set; } = 0.5;
        public double RangeAdjustmentWeight { get; set; } = 0.1;
        public DateTime LastUpdated { get; set; } = DateTime.Now;
        public int UpdateCount { get; set; } = 0;
    }

    public class EnsemblePredictionResult
    {
        public DateTime Date { get; set; }
        public double BayesianLow { get; set; }
        public double BayesianHigh { get; set; }
        public double LightGbmLow { get; set; }
        public double LightGbmHigh { get; set; }
        public double LightGbmRange { get; set; }
        public double FinalLow { get; set; }
        public double FinalHigh { get; set; }
        public EnsembleWeights Weights { get; set; }
        public double BayesianConfidence { get; set; }
        public string MarketRegime { get; set; }
    }

    public class ModelPerformanceMetrics
    {
        public string ModelName { get; set; }
        public double RecentMAPE { get; set; }
        public double RecentMAE { get; set; }
        public double DirectionalAccuracy { get; set; }
        public int SampleCount { get; set; }
        public DateTime LastEvaluated { get; set; }
        public double PerformanceScore { get; set; }
    }
}