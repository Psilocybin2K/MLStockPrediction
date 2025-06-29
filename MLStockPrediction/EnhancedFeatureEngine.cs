namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MLStockPrediction.Models;

    public class EnhancedFeatureEngine : MarketFeatureEngine
    {
        public List<EnhancedMarketFeatures> CreateEnhancedMarketFeaturesForEnsemble(
            Dictionary<string, List<StockData>> allStockData)
        {
            Console.WriteLine("🔧 Creating enhanced market features optimized for ensemble learning...");

            List<EnhancedMarketFeatures> enhancedFeatures = base.CreateEnhancedMarketFeatures(allStockData);

            if (enhancedFeatures.Count == 0)
            {
                return enhancedFeatures;
            }

            // Add additional features for LightGBM optimization
            Console.WriteLine("⚡ Adding LightGBM-optimized features...");

            this.AddMomentumIndicators(enhancedFeatures, allStockData);
            this.AddVolumePriceRelationships(enhancedFeatures, allStockData);
            this.AddVolatilityIndicators(enhancedFeatures, allStockData);
            this.AddLagFeatures(enhancedFeatures);
            this.AddInteractionFeatures(enhancedFeatures);

            Console.WriteLine($"✅ Enhanced feature engineering completed: {enhancedFeatures.Count} samples with ~80+ features");

            return enhancedFeatures;
        }

        private void AddMomentumIndicators(List<EnhancedMarketFeatures> features, Dictionary<string, List<StockData>> allStockData)
        {
            Console.WriteLine("📊 Adding momentum indicators...");

            List<StockData> msftData = allStockData["MSFT"].OrderBy(x => x.Date).ToList();
            List<StockData> dowData = allStockData["DOW"].OrderBy(x => x.Date).ToList();
            List<StockData> qqqData = allStockData["QQQ"].OrderBy(x => x.Date).ToList();

            for (int i = 0; i < features.Count; i++)
            {
                EnhancedMarketFeatures feature = features[i];

                // Find corresponding data points
                StockData? msft = msftData.FirstOrDefault(x => x.Date == feature.Date);
                StockData? dow = dowData.FirstOrDefault(x => x.Date == feature.Date);
                StockData? qqq = qqqData.FirstOrDefault(x => x.Date == feature.Date);

                if (msft == null || dow == null || qqq == null)
                {
                    continue;
                }

                // RSI approximation (14-period)
                feature.MsftRSI = this.CalculateRSIApproximation(msftData, i + 1, 14);
                feature.DowRSI = this.CalculateRSIApproximation(dowData, i + 1, 14);
                feature.QqqRSI = this.CalculateRSIApproximation(qqqData, i + 1, 14);

                // MACD approximation
                feature.MsftMACD = this.CalculateMACDApproximation(msftData, i + 1);

                // Price momentum (20-day)
                feature.MsftMomentum20 = this.CalculateMomentum(msftData, i + 1, 20);
                feature.DowMomentum20 = this.CalculateMomentum(dowData, i + 1, 20);

                // Stochastic approximation
                feature.MsftStochastic = this.CalculateStochasticApproximation(msftData, i + 1, 14);
            }
        }

        private void AddVolumePriceRelationships(List<EnhancedMarketFeatures> features, Dictionary<string, List<StockData>> allStockData)
        {
            Console.WriteLine("💹 Adding volume-price relationships...");

            List<StockData> msftData = allStockData["MSFT"].OrderBy(x => x.Date).ToList();

            for (int i = 0; i < features.Count; i++)
            {
                EnhancedMarketFeatures feature = features[i];

                // On-Balance Volume approximation
                feature.MsftOBV = this.CalculateOBVApproximation(msftData, i + 1, 20);

                // Volume-Weighted Average Price
                feature.MsftVWAP = this.CalculateVWAP(msftData, i + 1, 20);

                // Volume Rate of Change
                feature.MsftVolumeROC = this.CalculateVolumeROC(msftData, i + 1, 10);

                // Price-Volume Trend
                feature.MsftPVT = this.CalculatePVT(msftData, i + 1, 20);
            }
        }

        private void AddVolatilityIndicators(List<EnhancedMarketFeatures> features, Dictionary<string, List<StockData>> allStockData)
        {
            Console.WriteLine("📈 Adding volatility indicators...");

            List<StockData> msftData = allStockData["MSFT"].OrderBy(x => x.Date).ToList();

            for (int i = 0; i < features.Count; i++)
            {
                EnhancedMarketFeatures feature = features[i];

                // Bollinger Band squeeze
                feature.MsftBBSqueeze = this.CalculateBollingerBandSqueeze(msftData, i + 1, 20);

                // Volatility ratio (current vs average)
                feature.MsftVolatilityRatio = this.CalculateVolatilityRatio(msftData, i + 1, 20);

                // True Range
                feature.MsftTrueRange = this.CalculateTrueRange(msftData, i + 1);

                // Volatility breakout indicator
                feature.MsftVolBreakout = this.CalculateVolatilityBreakout(msftData, i + 1, 20);
            }
        }

        private void AddLagFeatures(List<EnhancedMarketFeatures> features)
        {
            Console.WriteLine("⏱️ Adding lag features...");

            for (int i = 0; i < features.Count; i++)
            {
                EnhancedMarketFeatures feature = features[i];

                // 1-day lag features
                if (i >= 1)
                {
                    EnhancedMarketFeatures prev1 = features[i - 1];
                    feature.MsftReturn_Lag1 = prev1.MsftReturn;
                    feature.MsftVolatility_Lag1 = prev1.MsftVolatility;
                    feature.DowReturn_Lag1 = prev1.DowReturn;
                }

                // 2-day lag features
                if (i >= 2)
                {
                    EnhancedMarketFeatures prev2 = features[i - 2];
                    feature.MsftReturn_Lag2 = prev2.MsftReturn;
                    feature.DowReturn_Lag2 = prev2.DowReturn;
                }

                // 5-day lag features
                if (i >= 5)
                {
                    EnhancedMarketFeatures prev5 = features[i - 5];
                    feature.MsftReturn_Lag5 = prev5.MsftReturn;
                    feature.QqqReturn_Lag5 = prev5.QqqReturn;
                }

                // Rolling differences
                if (i >= 10)
                {
                    feature.MsftSMA5_Diff = feature.MsftSMA5 - features[i - 10].MsftSMA5;
                    feature.MsftSMA20_Diff = feature.MsftSMA20 - features[i - 10].MsftSMA20;
                }
            }
        }

        private void AddInteractionFeatures(List<EnhancedMarketFeatures> features)
        {
            Console.WriteLine("🔗 Adding interaction features...");

            foreach (EnhancedMarketFeatures feature in features)
            {
                // Cross-asset momentum
                feature.DowMsftMomentumRatio = SafeDivision(feature.DowROC10, feature.MsftROC10);
                feature.QqqMsftMomentumRatio = SafeDivision(feature.QqqROC10, feature.MsftROC10);

                // Volume-volatility interaction
                feature.MsftVolumeVolatilityProduct = feature.MsftVolume * feature.MsftVolatility;

                // Price position interactions
                feature.MsftDowPricePositionDiff = feature.MsftPricePosition - feature.DowPricePosition;
                feature.MsftQqqPricePositionDiff = feature.MsftPricePosition - feature.QqqPricePosition;

                // SMA cross signals
                feature.MsftSMA5_SMA20_Ratio = SafeDivision(feature.MsftSMA5, feature.MsftSMA20);
                feature.MsftSMA10_SMA20_Ratio = SafeDivision(feature.MsftSMA10, feature.MsftSMA20);

                // Volatility regime indicator
                feature.IsHighVolatilityRegime = feature.MsftVolatility20 > 0.025 ? 1.0 : 0.0;
                feature.IsLowVolumeRegime = feature.MsftVolume < 0.8 ? 1.0 : 0.0;

                // Momentum regime
                feature.IsStrongUptrend = feature.MsftROC10 > 0.02 ? 1.0 : 0.0;
                feature.IsStrongDowntrend = feature.MsftROC10 < -0.02 ? 1.0 : 0.0;
            }
        }

        // Technical indicator calculation methods
        private double CalculateRSIApproximation(List<StockData> data, int endIndex, int period)
        {
            if (endIndex < period + 1)
            {
                return 50.0; // Neutral RSI
            }

            double[] prices = data.Take(endIndex).Select(x => (double)x.Close).ToArray();
            double gains = 0, losses = 0;

            for (int i = Math.Max(0, prices.Length - period); i < prices.Length - 1; i++)
            {
                double change = prices[i + 1] - prices[i];
                if (change > 0)
                {
                    gains += change;
                }
                else
                {
                    losses -= change;
                }
            }

            double avgGain = gains / period;
            double avgLoss = losses / period;

            if (avgLoss == 0)
            {
                return 100.0;
            }

            double rs = avgGain / avgLoss;
            return 100.0 - (100.0 / (1.0 + rs));
        }

        private double CalculateMACDApproximation(List<StockData> data, int endIndex)
        {
            if (endIndex < 26)
            {
                return 0.0;
            }

            List<double> prices = data.Take(endIndex).Select(x => (double)x.Close).ToList();

            double ema12 = TechnicalIndicators.ExponentialMovingAverage(prices.Select(x => (decimal)x).ToList(), 12);
            double ema26 = TechnicalIndicators.ExponentialMovingAverage(prices.Select(x => (decimal)x).ToList(), 26);

            return ema12 - ema26;
        }

        private double CalculateMomentum(List<StockData> data, int endIndex, int period)
        {
            if (endIndex < period + 1)
            {
                return 0.0;
            }

            double current = (double)data[endIndex - 1].Close;
            double previous = (double)data[endIndex - 1 - period].Close;

            return previous == 0 ? 0.0 : (current - previous) / previous;
        }

        private double CalculateStochasticApproximation(List<StockData> data, int endIndex, int period)
        {
            if (endIndex < period)
            {
                return 50.0;
            }

            List<StockData> recentData = data.Skip(Math.Max(0, endIndex - period)).Take(period).ToList();
            double highest = (double)recentData.Max(x => x.High);
            double lowest = (double)recentData.Min(x => x.Low);
            double current = (double)data[endIndex - 1].Close;

            return highest == lowest ? 50.0 : ((current - lowest) / (highest - lowest)) * 100.0;
        }

        private double CalculateOBVApproximation(List<StockData> data, int endIndex, int period)
        {
            if (endIndex < 2)
            {
                return 0.0;
            }

            double obv = 0;
            for (int i = Math.Max(1, endIndex - period); i < endIndex; i++)
            {
                if (data[i].Close > data[i - 1].Close)
                {
                    obv += data[i].Volume;
                }
                else if (data[i].Close < data[i - 1].Close)
                {
                    obv -= data[i].Volume;
                }
            }

            return obv / 1000000.0; // Scale down
        }

        private double CalculateVWAP(List<StockData> data, int endIndex, int period)
        {
            if (endIndex < period)
            {
                return (double)data[Math.Max(0, endIndex - 1)].Close;
            }

            double totalPriceVolume = 0;
            double totalVolume = 0;

            for (int i = Math.Max(0, endIndex - period); i < endIndex; i++)
            {
                double typicalPrice = (double)(data[i].High + data[i].Low + data[i].Close) / 3.0;
                totalPriceVolume += typicalPrice * data[i].Volume;
                totalVolume += data[i].Volume;
            }

            return totalVolume == 0 ? (double)data[endIndex - 1].Close : totalPriceVolume / totalVolume;
        }

        private double CalculateVolumeROC(List<StockData> data, int endIndex, int period)
        {
            if (endIndex < period + 1)
            {
                return 0.0;
            }

            double current = data[endIndex - 1].Volume;
            double previous = data[endIndex - 1 - period].Volume;

            return previous == 0 ? 0.0 : (current - previous) / previous;
        }

        private double CalculatePVT(List<StockData> data, int endIndex, int period)
        {
            if (endIndex < 2)
            {
                return 0.0;
            }

            double pvt = 0;
            for (int i = Math.Max(1, endIndex - period); i < endIndex; i++)
            {
                double priceChange = (double)(data[i].Close - data[i - 1].Close);
                double priceChangePercent = (double)data[i - 1].Close == 0 ? 0 : priceChange / (double)data[i - 1].Close;
                pvt += priceChangePercent * data[i].Volume;
            }

            return pvt / 1000000.0; // Scale down
        }

        private double CalculateBollingerBandSqueeze(List<StockData> data, int endIndex, int period)
        {
            if (endIndex < period)
            {
                return 0.0;
            }

            List<double> prices = data.Skip(Math.Max(0, endIndex - period)).Take(period).Select(x => (double)x.Close).ToList();
            double sma = prices.Average();
            double stdDev = Math.Sqrt(prices.Select(x => Math.Pow(x - sma, 2)).Average());

            return stdDev / sma; // Relative standard deviation
        }

        private double CalculateVolatilityRatio(List<StockData> data, int endIndex, int period)
        {
            if (endIndex < period)
            {
                return 1.0;
            }

            List<StockData> recentData = data.Skip(Math.Max(0, endIndex - 5)).Take(5).ToList();
            List<StockData> historicalData = data.Skip(Math.Max(0, endIndex - period)).Take(period).ToList();

            double recentVol = recentData.Average(x => (double)(x.High - x.Low) / (double)x.Close);
            double historicalVol = historicalData.Average(x => (double)(x.High - x.Low) / (double)x.Close);

            return historicalVol == 0 ? 1.0 : recentVol / historicalVol;
        }

        private double CalculateTrueRange(List<StockData> data, int endIndex)
        {
            if (endIndex < 2)
            {
                return 0.0;
            }

            StockData current = data[endIndex - 1];
            StockData previous = data[endIndex - 2];

            double tr1 = (double)(current.High - current.Low);
            double tr2 = Math.Abs((double)(current.High - previous.Close));
            double tr3 = Math.Abs((double)(current.Low - previous.Close));

            return Math.Max(tr1, Math.Max(tr2, tr3)) / (double)current.Close;
        }

        private double CalculateVolatilityBreakout(List<StockData> data, int endIndex, int period)
        {
            if (endIndex < period)
            {
                return 0.0;
            }

            List<StockData> recentData = data.Skip(Math.Max(0, endIndex - period)).Take(period).ToList();
            double avgVolatility = recentData.Average(x => (double)(x.High - x.Low) / (double)x.Close);
            double currentVolatility = (double)(data[endIndex - 1].High - data[endIndex - 1].Low) / (double)data[endIndex - 1].Close;

            return avgVolatility == 0 ? 0.0 : (currentVolatility - avgVolatility) / avgVolatility;
        }

        private static double SafeDivision(double numerator, double denominator)
        {
            return Math.Abs(denominator) < 1e-8 ? 0.0 : numerator / denominator;
        }
    }
}