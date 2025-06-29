namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MLStockPrediction.Models;

    public class EnhancedMarketFeatureEngine : MarketFeatureEngine
    {
        public new List<EnhancedMarketFeatures> CreateMarketFeatures(
            Dictionary<string, List<StockData>> allStockData)
        {
            Console.WriteLine("🔧 Creating enhanced market features...");
            List<EnhancedMarketFeatures> features = new List<EnhancedMarketFeatures>();

            if (!allStockData.ContainsKey("DOW") ||
                !allStockData.ContainsKey("QQQ") ||
                !allStockData.ContainsKey("MSFT"))
            {
                Console.WriteLine("❌ Missing required stock data");
                return features;
            }

            List<StockData> dowData = allStockData["DOW"].OrderBy(x => x.Date).ToList();
            List<StockData> qqqData = allStockData["QQQ"].OrderBy(x => x.Date).ToList();
            List<StockData> msftData = allStockData["MSFT"].OrderBy(x => x.Date).ToList();

            // Find common dates
            List<DateTime> commonDates = dowData.Select(x => x.Date)
                .Intersect(qqqData.Select(x => x.Date))
                .Intersect(msftData.Select(x => x.Date))
                .OrderBy(x => x)
                .ToList();

            Console.WriteLine($"Found {commonDates.Count} common dates");

            for (int i = 1; i < commonDates.Count; i++)
            {
                DateTime date = commonDates[i];
                DateTime prevDate = commonDates[i - 1];

                // Get current and historical data up to this point
                List<StockData> dowHistory = dowData.Where(x => x.Date <= date).OrderBy(x => x.Date).ToList();
                List<StockData> qqqHistory = qqqData.Where(x => x.Date <= date).OrderBy(x => x.Date).ToList();
                List<StockData> msftHistory = msftData.Where(x => x.Date <= date).OrderBy(x => x.Date).ToList();

                StockData dow = dowHistory.Last();
                StockData dowPrev = dowData.First(x => x.Date == prevDate);
                StockData qqq = qqqHistory.Last();
                StockData qqqPrev = qqqData.First(x => x.Date == prevDate);
                StockData msft = msftHistory.Last();
                StockData msftPrev = msftData.First(x => x.Date == prevDate);

                // Calculate basic features (from base class)
                EnhancedMarketFeatures feature = new EnhancedMarketFeatures
                {
                    Date = date,
                    DowReturn = this.CalculateReturn(dowPrev.Close, dow.Close),
                    DowVolatility = this.CalculateVolatility(dow),
                    DowVolume = this.NormalizeVolume(dow.Volume, dowData),
                    QqqReturn = this.CalculateReturn(qqqPrev.Close, qqq.Close),
                    QqqVolatility = this.CalculateVolatility(qqq),
                    QqqVolume = this.NormalizeVolume(qqq.Volume, qqqData),
                    MsftReturn = this.CalculateReturn(msftPrev.Close, msft.Close),
                    MsftVolatility = this.CalculateVolatility(msft),
                    MsftVolume = this.NormalizeVolume(msft.Volume, msftData),
                    DowMsftCorrelation = this.CalculateCorrelation(dowData, msftData, i, 10),
                    QqqMsftCorrelation = this.CalculateCorrelation(qqqData, msftData, i, 10),
                    MsftLow = (double)msft.Low,
                    MsftHigh = (double)msft.High
                };

                // Calculate enhanced technical features
                this.CalculateEnhancedFeatures(feature, dowHistory, qqqHistory, msftHistory);

                features.Add(feature);

                // Log first few features for debugging
                if (i <= 3)
                {
                    Console.WriteLine($"Enhanced Feature {i}: Date={date:yyyy-MM-dd}");
                    Console.WriteLine($"   MSFT SMA20={feature.MsftSMA20:F2}, EMA Ratio={feature.MsftEMAR20:F4}");
                    Console.WriteLine($"   Price Position={feature.MsftPricePosition:F4}, ATR={feature.MsftATR:F2}");
                }
            }

            Console.WriteLine($"✅ Created {features.Count} enhanced market features");
            return features;
        }

        private void CalculateEnhancedFeatures(
            EnhancedMarketFeatures feature,
            List<StockData> dowHistory,
            List<StockData> qqqHistory,
            List<StockData> msftHistory)
        {
            // Extract price lists
            List<decimal> dowPrices = dowHistory.Select(x => x.Close).ToList();
            List<decimal> qqqPrices = qqqHistory.Select(x => x.Close).ToList();
            List<decimal> msftPrices = msftHistory.Select(x => x.Close).ToList();

            List<double> dowReturns = this.CalculateReturnSeries(dowHistory);
            List<double> qqqReturns = this.CalculateReturnSeries(qqqHistory);
            List<double> msftReturns = this.CalculateReturnSeries(msftHistory);

            // Moving Averages
            feature.DowSMA5 = TechnicalIndicators.SimpleMovingAverage(dowPrices, 5);
            feature.DowSMA10 = TechnicalIndicators.SimpleMovingAverage(dowPrices, 10);
            feature.DowSMA20 = TechnicalIndicators.SimpleMovingAverage(dowPrices, 20);
            feature.QqqSMA5 = TechnicalIndicators.SimpleMovingAverage(qqqPrices, 5);
            feature.QqqSMA10 = TechnicalIndicators.SimpleMovingAverage(qqqPrices, 10);
            feature.QqqSMA20 = TechnicalIndicators.SimpleMovingAverage(qqqPrices, 20);
            feature.MsftSMA5 = TechnicalIndicators.SimpleMovingAverage(msftPrices, 5);
            feature.MsftSMA10 = TechnicalIndicators.SimpleMovingAverage(msftPrices, 10);
            feature.MsftSMA20 = TechnicalIndicators.SimpleMovingAverage(msftPrices, 20);

            // EMA Ratios
            double dowEMA5 = TechnicalIndicators.ExponentialMovingAverage(dowPrices, 5);
            double dowEMA10 = TechnicalIndicators.ExponentialMovingAverage(dowPrices, 10);
            double dowEMA20 = TechnicalIndicators.ExponentialMovingAverage(dowPrices, 20);
            feature.DowEMAR5 = dowEMA5 > 0 ? (double)dowPrices.Last() / dowEMA5 : 1.0;
            feature.DowEMAR10 = dowEMA10 > 0 ? (double)dowPrices.Last() / dowEMA10 : 1.0;
            feature.DowEMAR20 = dowEMA20 > 0 ? (double)dowPrices.Last() / dowEMA20 : 1.0;

            double qqqEMA5 = TechnicalIndicators.ExponentialMovingAverage(qqqPrices, 5);
            double qqqEMA10 = TechnicalIndicators.ExponentialMovingAverage(qqqPrices, 10);
            double qqqEMA20 = TechnicalIndicators.ExponentialMovingAverage(qqqPrices, 20);
            feature.QqqEMAR5 = qqqEMA5 > 0 ? (double)qqqPrices.Last() / qqqEMA5 : 1.0;
            feature.QqqEMAR10 = qqqEMA10 > 0 ? (double)qqqPrices.Last() / qqqEMA10 : 1.0;
            feature.QqqEMAR20 = qqqEMA20 > 0 ? (double)qqqPrices.Last() / qqqEMA20 : 1.0;

            double msftEMA5 = TechnicalIndicators.ExponentialMovingAverage(msftPrices, 5);
            double msftEMA10 = TechnicalIndicators.ExponentialMovingAverage(msftPrices, 10);
            double msftEMA20 = TechnicalIndicators.ExponentialMovingAverage(msftPrices, 20);
            feature.MsftEMAR5 = msftEMA5 > 0 ? (double)msftPrices.Last() / msftEMA5 : 1.0;
            feature.MsftEMAR10 = msftEMA10 > 0 ? (double)msftPrices.Last() / msftEMA10 : 1.0;
            feature.MsftEMAR20 = msftEMA20 > 0 ? (double)msftPrices.Last() / msftEMA20 : 1.0;

            // Price Position in Range
            feature.DowPricePosition = TechnicalIndicators.PricePositionInRange(dowPrices.Last(), dowPrices);
            feature.QqqPricePosition = TechnicalIndicators.PricePositionInRange(qqqPrices.Last(), qqqPrices);
            feature.MsftPricePosition = TechnicalIndicators.PricePositionInRange(msftPrices.Last(), msftPrices);

            // Rate of Change (Momentum)
            feature.DowROC5 = TechnicalIndicators.RateOfChange(dowPrices, 5);
            feature.DowROC10 = TechnicalIndicators.RateOfChange(dowPrices, 10);
            feature.QqqROC5 = TechnicalIndicators.RateOfChange(qqqPrices, 5);
            feature.QqqROC10 = TechnicalIndicators.RateOfChange(qqqPrices, 10);
            feature.MsftROC5 = TechnicalIndicators.RateOfChange(msftPrices, 5);
            feature.MsftROC10 = TechnicalIndicators.RateOfChange(msftPrices, 10);

            // Rolling Volatility
            feature.DowVolatility5 = TechnicalIndicators.RollingStandardDeviation(dowReturns, 5);
            feature.DowVolatility10 = TechnicalIndicators.RollingStandardDeviation(dowReturns, 10);
            feature.DowVolatility20 = TechnicalIndicators.RollingStandardDeviation(dowReturns, 20);
            feature.QqqVolatility5 = TechnicalIndicators.RollingStandardDeviation(qqqReturns, 5);
            feature.QqqVolatility10 = TechnicalIndicators.RollingStandardDeviation(qqqReturns, 10);
            feature.QqqVolatility20 = TechnicalIndicators.RollingStandardDeviation(qqqReturns, 20);
            feature.MsftVolatility5 = TechnicalIndicators.RollingStandardDeviation(msftReturns, 5);
            feature.MsftVolatility10 = TechnicalIndicators.RollingStandardDeviation(msftReturns, 10);
            feature.MsftVolatility20 = TechnicalIndicators.RollingStandardDeviation(msftReturns, 20);

            // Average True Range
            feature.DowATR = TechnicalIndicators.AverageTrueRange(dowHistory);
            feature.QqqATR = TechnicalIndicators.AverageTrueRange(qqqHistory);
            feature.MsftATR = TechnicalIndicators.AverageTrueRange(msftHistory);

            // Volatility Ratios
            double dowAvgVol = dowReturns.Count > 50 ? TechnicalIndicators.RollingStandardDeviation(dowReturns, 50) : feature.DowVolatility20;
            double qqqAvgVol = qqqReturns.Count > 50 ? TechnicalIndicators.RollingStandardDeviation(qqqReturns, 50) : feature.QqqVolatility20;
            double msftAvgVol = msftReturns.Count > 50 ? TechnicalIndicators.RollingStandardDeviation(msftReturns, 50) : feature.MsftVolatility20;

            feature.DowVolatilityRatio = dowAvgVol > 0 ? feature.DowVolatility20 / dowAvgVol : 1.0;
            feature.QqqVolatilityRatio = qqqAvgVol > 0 ? feature.QqqVolatility20 / qqqAvgVol : 1.0;
            feature.MsftVolatilityRatio = msftAvgVol > 0 ? feature.MsftVolatility20 / msftAvgVol : 1.0;

            // Bollinger Band Positions
            feature.DowBBPosition = TechnicalIndicators.BollingerBandPosition(dowPrices.Last(), dowPrices);
            feature.QqqBBPosition = TechnicalIndicators.BollingerBandPosition(qqqPrices.Last(), qqqPrices);
            feature.MsftBBPosition = TechnicalIndicators.BollingerBandPosition(msftPrices.Last(), msftPrices);
        }

        private List<double> CalculateReturnSeries(List<StockData> data)
        {
            List<double> returns = new List<double>();

            for (int i = 1; i < data.Count; i++)
            {
                double ret = this.CalculateReturn(data[i - 1].Close, data[i].Close);
                returns.Add(ret);
            }

            return returns;
        }
    }
}