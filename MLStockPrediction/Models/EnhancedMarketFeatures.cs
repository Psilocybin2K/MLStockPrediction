namespace MLStockPrediction.Models
{
    // Extension of EnhancedMarketFeatures for ensemble learning
    public partial class EnhancedMarketFeatures
    {
        // Momentum Indicators
        public double MsftRSI { get; set; }
        public double DowRSI { get; set; }
        public double QqqRSI { get; set; }
        public double MsftMACD { get; set; }
        public double MsftMomentum20 { get; set; }
        public double DowMomentum20 { get; set; }
        public double MsftStochastic { get; set; }

        // Volume-Price Relationships  
        public double MsftOBV { get; set; }
        public double MsftVWAP { get; set; }
        public double MsftVolumeROC { get; set; }
        public double MsftPVT { get; set; }

        // Volatility Indicators
        public double MsftBBSqueeze { get; set; }
        public double MsftTrueRange { get; set; }
        public double MsftVolBreakout { get; set; }

        // Lag Features
        public double MsftReturn_Lag1 { get; set; }
        public double MsftReturn_Lag2 { get; set; }
        public double MsftReturn_Lag5 { get; set; }
        public double MsftVolatility_Lag1 { get; set; }
        public double DowReturn_Lag1 { get; set; }
        public double DowReturn_Lag2 { get; set; }
        public double QqqReturn_Lag5 { get; set; }

        // Rolling Differences
        public double MsftSMA5_Diff { get; set; }
        public double MsftSMA20_Diff { get; set; }

        // Interaction Features
        public double DowMsftMomentumRatio { get; set; }
        public double QqqMsftMomentumRatio { get; set; }
        public double MsftVolumeVolatilityProduct { get; set; }
        public double MsftDowPricePositionDiff { get; set; }
        public double MsftQqqPricePositionDiff { get; set; }
        public double MsftSMA5_SMA20_Ratio { get; set; }
        public double MsftSMA10_SMA20_Ratio { get; set; }

        // Regime Indicators
        public double IsHighVolatilityRegime { get; set; }
        public double IsLowVolumeRegime { get; set; }
        public double IsStrongUptrend { get; set; }
        public double IsStrongDowntrend { get; set; }
    }

    public partial class EnhancedMarketFeatures : MarketFeatures
    {
        // Moving Averages for each asset
        public double DowSMA5 { get; set; }
        public double DowSMA10 { get; set; }
        public double DowSMA20 { get; set; }
        public double QqqSMA5 { get; set; }
        public double QqqSMA10 { get; set; }
        public double QqqSMA20 { get; set; }
        public double MsftSMA5 { get; set; }
        public double MsftSMA10 { get; set; }
        public double MsftSMA20 { get; set; }

        // EMA Ratios (Current Price / EMA)
        public double DowEMAR5 { get; set; }
        public double DowEMAR10 { get; set; }
        public double DowEMAR20 { get; set; }
        public double QqqEMAR5 { get; set; }
        public double QqqEMAR10 { get; set; }
        public double QqqEMAR20 { get; set; }
        public double MsftEMAR5 { get; set; }
        public double MsftEMAR10 { get; set; }
        public double MsftEMAR20 { get; set; }

        // Price Position in Range
        public double DowPricePosition { get; set; }
        public double QqqPricePosition { get; set; }
        public double MsftPricePosition { get; set; }

        // Rate of Change (Momentum)
        public double DowROC5 { get; set; }
        public double DowROC10 { get; set; }
        public double QqqROC5 { get; set; }
        public double QqqROC10 { get; set; }
        public double MsftROC5 { get; set; }
        public double MsftROC10 { get; set; }

        // Rolling Volatility
        public double DowVolatility5 { get; set; }
        public double DowVolatility10 { get; set; }
        public double DowVolatility20 { get; set; }
        public double QqqVolatility5 { get; set; }
        public double QqqVolatility10 { get; set; }
        public double QqqVolatility20 { get; set; }
        public double MsftVolatility5 { get; set; }
        public double MsftVolatility10 { get; set; }
        public double MsftVolatility20 { get; set; }

        // Average True Range
        public double DowATR { get; set; }
        public double QqqATR { get; set; }
        public double MsftATR { get; set; }

        // Volatility Ratios (Current / Historical Average)
        public double DowVolatilityRatio { get; set; }
        public double QqqVolatilityRatio { get; set; }
        public double MsftVolatilityRatio { get; set; }

        // Bollinger Band Positions
        public double DowBBPosition { get; set; }
        public double QqqBBPosition { get; set; }
        public double MsftBBPosition { get; set; }

        // NEW: Temporal & Cyclical Features

        // Day of Week Effects (0-1 encoding)
        public double IsMondayEffect { get; set; }
        public double IsTuesdayEffect { get; set; }
        public double IsWednesdayEffect { get; set; }
        public double IsThursdayEffect { get; set; }
        public double IsFridayEffect { get; set; }

        // Week of Month Patterns
        public double IsFirstWeekOfMonth { get; set; }
        public double IsSecondWeekOfMonth { get; set; }
        public double IsThirdWeekOfMonth { get; set; }
        public double IsFourthWeekOfMonth { get; set; }
        public double IsOptionsExpirationWeek { get; set; }

        // Month Effects
        public double IsJanuaryEffect { get; set; }
        public double IsQuarterStart { get; set; }
        public double IsQuarterEnd { get; set; }
        public double IsYearEnd { get; set; }

        // Holiday & Market Cycle Proximity
        public double DaysToMarketHoliday { get; set; }
        public double DaysFromMarketHoliday { get; set; }
        public double DaysIntoQuarter { get; set; }
        public double DaysUntilQuarterEnd { get; set; }

        // Earnings Season Indicators
        public double IsEarningsSeason { get; set; }
        public double DaysToEarningsWeek { get; set; }
        public double DaysFromEarningsWeek { get; set; }

        // Seasonal Adjustments
        public double QuarterProgress { get; set; } // 0-1 through quarter
        public double YearProgress { get; set; }    // 0-1 through year
        public double MonthProgress { get; set; }   // 0-1 through month
    }
}