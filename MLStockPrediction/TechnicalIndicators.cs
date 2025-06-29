namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MLStockPrediction.Models;

    public class TechnicalIndicators
    {
        public static double SimpleMovingAverage(List<decimal> prices, int period)
        {
            return prices.Count < period ? (double)prices.LastOrDefault() : (double)prices.TakeLast(period).Average();
        }

        public static double ExponentialMovingAverage(List<decimal> prices, int period)
        {
            if (prices.Count == 0)
            {
                return 0;
            }

            if (prices.Count < period)
            {
                return (double)prices.Average();
            }

            double multiplier = 2.0 / (period + 1);
            double ema = (double)prices.First();

            for (int i = 1; i < prices.Count; i++)
            {
                ema = ((double)prices[i] * multiplier) + (ema * (1 - multiplier));
            }

            return ema;
        }

        public static double PricePositionInRange(decimal currentPrice, List<decimal> recentPrices, int period = 20)
        {
            if (recentPrices.Count < period)
            {
                return 0.5;
            }

            List<decimal> recent = recentPrices.TakeLast(period).ToList();
            double min = (double)recent.Min();
            double max = (double)recent.Max();

            return max == min ? 0.5 : ((double)currentPrice - min) / (max - min);
        }

        public static double RateOfChange(List<decimal> prices, int period)
        {
            if (prices.Count <= period)
            {
                return 0;
            }

            double current = (double)prices.Last();
            double previous = (double)prices[prices.Count - 1 - period];

            return previous == 0 ? 0 : (current - previous) / previous;
        }

        public static double RollingStandardDeviation(List<double> values, int period)
        {
            if (values.Count < period)
            {
                return 0;
            }

            List<double> recent = values.TakeLast(period).ToList();
            double mean = recent.Average();
            double sumSquaredDiffs = recent.Sum(x => Math.Pow(x - mean, 2));

            return Math.Sqrt(sumSquaredDiffs / period);
        }

        public static double AverageTrueRange(List<StockData> data, int period = 14)
        {
            if (data.Count < 2)
            {
                return 0;
            }

            List<double> trueRanges = new List<double>();

            for (int i = 1; i < data.Count; i++)
            {
                StockData current = data[i];
                StockData previous = data[i - 1];

                double tr1 = (double)(current.High - current.Low);
                double tr2 = Math.Abs((double)(current.High - previous.Close));
                double tr3 = Math.Abs((double)(current.Low - previous.Close));

                trueRanges.Add(Math.Max(tr1, Math.Max(tr2, tr3)));
            }

            return trueRanges.Count < period ? trueRanges.Average() : trueRanges.TakeLast(period).Average();
        }

        public static double BollingerBandPosition(decimal currentPrice, List<decimal> prices, int period = 20)
        {
            if (prices.Count < period)
            {
                return 0;
            }

            double sma = SimpleMovingAverage(prices, period);
            List<double> recent = prices.TakeLast(period).Select(x => (double)x).ToList();
            double stdDev = RollingStandardDeviation(recent, period);

            return stdDev == 0 ? 0 : ((double)currentPrice - sma) / (2 * stdDev);
        }
    }
}