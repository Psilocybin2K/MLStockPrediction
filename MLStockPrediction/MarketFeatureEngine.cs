namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MLStockPrediction.Models;

    public class MarketFeatureEngine
    {
        public List<MarketFeatures> CreateMarketFeatures(
            Dictionary<string, List<StockData>> allStockData)
        {
            Console.WriteLine("🔧 Creating market features...");
            List<MarketFeatures> features = new List<MarketFeatures>();

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

                StockData dow = dowData.First(x => x.Date == date);
                StockData dowPrev = dowData.First(x => x.Date == prevDate);

                StockData qqq = qqqData.First(x => x.Date == date);
                StockData qqqPrev = qqqData.First(x => x.Date == prevDate);

                StockData msft = msftData.First(x => x.Date == date);
                StockData msftPrev = msftData.First(x => x.Date == prevDate);

                MarketFeatures feature = new MarketFeatures
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

                features.Add(feature);

                // Log first few features for debugging
                if (i <= 3)
                {
                    Console.WriteLine($"Feature {i}: Date={date:yyyy-MM-dd}, MSFT Low={feature.MsftLow:F2}, High={feature.MsftHigh:F2}");
                    Console.WriteLine($"   Returns: DOW={feature.DowReturn:F4}, QQQ={feature.QqqReturn:F4}, MSFT={feature.MsftReturn:F4}");
                }
            }

            Console.WriteLine($"✅ Created {features.Count} market features");
            return features;
        }

        protected double CalculateReturn(decimal prev, decimal current)
        {
            return prev <= 0 ? 0 : (double)((current - prev) / prev);
        }

        protected double CalculateVolatility(StockData data)
        {
            return data.Close <= 0 ? 0 : (double)((data.High - data.Low) / data.Close);
        }

        protected double NormalizeVolume(long volume, List<StockData> allData)
        {
            double avgVolume = allData.Average(x => x.Volume);
            return avgVolume <= 0 ? 1.0 : volume / avgVolume;
        }

        protected double CalculateCorrelation(List<StockData> data1, List<StockData> data2, int currentIndex, int window)
        {
            // Simple rolling correlation based on returns
            if (currentIndex < window)
            {
                return 0.5; // Default correlation
            }

            List<double> returns1 = new List<double>();
            List<double> returns2 = new List<double>();

            for (int i = Math.Max(1, currentIndex - window); i < currentIndex; i++)
            {
                if (i < data1.Count && i < data2.Count)
                {
                    double ret1 = this.CalculateReturn(data1[i - 1].Close, data1[i].Close);
                    double ret2 = this.CalculateReturn(data2[i - 1].Close, data2[i].Close);
                    returns1.Add(ret1);
                    returns2.Add(ret2);
                }
            }

            if (returns1.Count < 2)
            {
                return 0.5;
            }

            // Calculate Pearson correlation
            double mean1 = returns1.Average();
            double mean2 = returns2.Average();

            double numerator = 0;
            double sumSq1 = 0;
            double sumSq2 = 0;

            for (int i = 0; i < returns1.Count; i++)
            {
                double diff1 = returns1[i] - mean1;
                double diff2 = returns2[i] - mean2;
                numerator += diff1 * diff2;
                sumSq1 += diff1 * diff1;
                sumSq2 += diff2 * diff2;
            }

            double denominator = Math.Sqrt(sumSq1 * sumSq2);
            return denominator < 1e-8 ? 0.5 : numerator / denominator;
        }
    }
}