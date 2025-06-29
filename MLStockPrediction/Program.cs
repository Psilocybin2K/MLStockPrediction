namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;

    using CsvHelper;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Models;

    using MLStockPrediction.Converters;
    using MLStockPrediction.Models;

    using Range = Microsoft.ML.Probabilistic.Models.Range;

    public class BayesianStockModel
    {
        private readonly InferenceEngine _engine;
        private Gaussian[] _lowWeights;
        private Gaussian _lowBias;
        private Gamma _lowPrecision;
        private Gaussian[] _highWeights;
        private Gaussian _highBias;
        private Gamma _highPrecision;
        private bool _isTrained = false;
        private double _targetMean = 0;
        private double _targetStd = 1;

        public BayesianStockModel()
        {
            this._engine = new InferenceEngine();
        }

        public void Train(List<MarketFeatures> trainingData)
        {
            if (trainingData.Count < 2)
            {
                return;
            }

            int n = trainingData.Count;

            // Extract features and targets
            double[,] features = this.ExtractFeatureMatrix(trainingData);
            double[] lowTargets = trainingData.Select(x => x.MsftLow).ToArray();
            double[] highTargets = trainingData.Select(x => x.MsftHigh).ToArray();

            // Normalize targets to help Bayesian inference
            this._targetMean = lowTargets.Concat(highTargets).Average();
            this._targetStd = Math.Sqrt(lowTargets.Concat(highTargets).Select(x => Math.Pow(x - this._targetMean, 2)).Average());

            double[] normalizedLowTargets = lowTargets.Select(x => (x - this._targetMean) / this._targetStd).ToArray();
            double[] normalizedHighTargets = highTargets.Select(x => (x - this._targetMean) / this._targetStd).ToArray();

            // Train separate models for Low and High predictions
            this.TrainPriceModel(features, normalizedLowTargets, true);
            this.TrainPriceModel(features, normalizedHighTargets, false);
        }

        private void TrainPriceModel(double[,] features, double[] targets, bool isLowModel)
        {
            int n = features.GetLength(0);
            int numFeatures = features.GetLength(1);

            // Define ranges
            Range dataRange = new Range(n);
            Range featureRange = new Range(numFeatures);

            // More informative priors
            VariableArray<double> weights = Variable.Array<double>(featureRange);
            weights[featureRange] = Variable.GaussianFromMeanAndVariance(0, 0.1).ForEach(featureRange);

            Variable<double> bias = Variable.GaussianFromMeanAndVariance(0, 1.0);
            Variable<double> precision = Variable.GammaFromShapeAndRate(2, 1);

            // Data
            VariableArray2D<double> featuresVar = Variable.Array<double>(dataRange, featureRange);
            VariableArray<double> targetsVar = Variable.Array<double>(dataRange);

            // Model
            using (Variable.ForEach(dataRange))
            {
                VariableArray<double> products = Variable.Array<double>(featureRange);
                products[featureRange] = featuresVar[dataRange, featureRange] * weights[featureRange];
                Variable<double> dotProduct = Variable.Sum(products);
                targetsVar[dataRange] = Variable.GaussianFromMeanAndPrecision(bias + dotProduct, precision);
            }

            // Set observed data
            featuresVar.ObservedValue = features;
            targetsVar.ObservedValue = targets;

            // Inference
            Gaussian[] weights_post = this._engine.Infer<Gaussian[]>(weights);
            Gaussian bias_post = this._engine.Infer<Gaussian>(bias);
            Gamma precision_post = this._engine.Infer<Gamma>(precision);

            if (isLowModel)
            {
                this._lowWeights = weights_post;
                this._lowBias = bias_post;
                this._lowPrecision = precision_post;
            }
            else
            {
                this._highWeights = weights_post;
                this._highBias = bias_post;
                this._highPrecision = precision_post;
            }

            this._isTrained = true;
        }

        public (double Low, double High) Predict(MarketFeatures marketData)
        {
            if (!this._isTrained)
            {
                throw new InvalidOperationException("Model must be trained first");
            }

            double[] featureVector = this.ExtractFeatureVector(marketData);

            // Predict Low
            double lowPredictionNorm = this._lowBias.GetMean();
            for (int i = 0; i < featureVector.Length && i < this._lowWeights.Length; i++)
            {
                lowPredictionNorm += featureVector[i] * this._lowWeights[i].GetMean();
            }
            double lowPrediction = (lowPredictionNorm * this._targetStd) + this._targetMean;

            // Predict High
            double highPredictionNorm = this._highBias.GetMean();
            for (int i = 0; i < featureVector.Length && i < this._highWeights.Length; i++)
            {
                highPredictionNorm += featureVector[i] * this._highWeights[i].GetMean();
            }
            double highPrediction = (highPredictionNorm * this._targetStd) + this._targetMean;

            // Ensure high >= low
            if (highPrediction < lowPrediction)
            {
                double temp = highPrediction;
                highPrediction = lowPrediction;
                lowPrediction = temp;
            }

            return (lowPrediction, highPrediction);
        }

        private double[,] ExtractFeatureMatrix(List<MarketFeatures> data)
        {
            int n = data.Count;
            int numFeatures = 9;
            double[,] matrix = new double[n, numFeatures];

            for (int i = 0; i < n; i++)
            {
                MarketFeatures item = data[i];
                matrix[i, 0] = item.DowReturn;
                matrix[i, 1] = item.DowVolatility;
                matrix[i, 2] = Math.Tanh(item.DowVolume); // Bounded normalization
                matrix[i, 3] = item.QqqReturn;
                matrix[i, 4] = item.QqqVolatility;
                matrix[i, 5] = Math.Tanh(item.QqqVolume); // Bounded normalization
                matrix[i, 6] = item.MsftReturn;
                matrix[i, 7] = item.DowMsftCorrelation;
                matrix[i, 8] = item.QqqMsftCorrelation;
            }

            return matrix;
        }

        private double[] ExtractFeatureVector(MarketFeatures data)
        {
            return new double[]
            {
                data.DowReturn,
                data.DowVolatility,
                Math.Tanh(data.DowVolume),
                data.QqqReturn,
                data.QqqVolatility,
                Math.Tanh(data.QqqVolume),
                data.MsftReturn,
                data.DowMsftCorrelation,
                data.QqqMsftCorrelation
            };
        }
    }

    public class MarketFeatureEngine
    {
        public List<MarketFeatures> CreateMarketFeatures(
            Dictionary<string, List<StockData>> allStockData)
        {
            List<MarketFeatures> features = new List<MarketFeatures>();

            if (!allStockData.ContainsKey("DOW") ||
                !allStockData.ContainsKey("QQQ") ||
                !allStockData.ContainsKey("MSFT"))
            {
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
                    DowMsftCorrelation = 0.7, // Simplified - would calculate rolling correlation
                    QqqMsftCorrelation = 0.8, // Simplified - would calculate rolling correlation
                    MsftLow = (double)msft.Low,
                    MsftHigh = (double)msft.High
                };

                features.Add(feature);
            }

            return features;
        }

        private double CalculateReturn(decimal prev, decimal current)
        {
            return (double)((current - prev) / prev);
        }

        private double CalculateVolatility(StockData data)
        {
            return (double)((data.High - data.Low) / data.Close);
        }

        private double NormalizeVolume(long volume, List<StockData> allData)
        {
            double avgVolume = allData.Average(x => x.Volume);
            return volume / (double)avgVolume;
        }
    }

    public class StockDataLoader
    {
        public async Task<Dictionary<string, List<StockData>>> LoadAllStockDataAsync()
        {
            string[] stockFiles = new[] { "DOW.csv", "QQQ.csv", "MSFT.csv" };
            Dictionary<string, List<StockData>> stockData = new Dictionary<string, List<StockData>>();

            foreach (string file in stockFiles)
            {
                string symbol = Path.GetFileNameWithoutExtension(file);
                List<StockData> data = await this.LoadStockDataFromFileAsync(file);
                stockData[symbol] = data;
            }

            return stockData;
        }

        public async Task<List<StockData>> LoadStockDataFromFileAsync(string filePath)
        {
            string csvContent = await File.ReadAllTextAsync(filePath);
            return this.ParseCsv(csvContent);
        }

        public List<StockData> ParseCsv(string csvContent)
        {
            using StringReader reader = new StringReader(csvContent);
            using CsvReader csv = new CsvReader(reader, CultureInfo.InvariantCulture);

            csv.Context.TypeConverterCache.AddConverter<decimal>(new DecimalConverter());
            return csv.GetRecords<StockData>().ToList();
        }

        public void DisplayStockSummary(Dictionary<string, List<StockData>> allStockData)
        {
            foreach ((string symbol, List<StockData> data) in allStockData)
            {
                Console.WriteLine($"\n=== {symbol} Stock Data ===");
                Console.WriteLine($"Records: {data.Count}");

                if (data.Any())
                {
                    List<StockData> orderedData = data.OrderBy(x => x.Date).ToList();
                    StockData latest = orderedData.Last();
                    StockData earliest = orderedData.First();

                    Console.WriteLine($"Date Range: {earliest.Date:yyyy-MM-dd} to {latest.Date:yyyy-MM-dd}");
                    Console.WriteLine($"Latest Close: ${latest.Close:F2}");
                    Console.WriteLine($"Price Range: ${orderedData.Min(x => x.Low):F2} - ${orderedData.Max(x => x.High):F2}");
                    Console.WriteLine($"Avg Volume: {data.Average(x => x.Volume):N0}");

                    Console.WriteLine("\nRecent 3 days:");
                    foreach (StockData? record in orderedData.TakeLast(3))
                    {
                        Console.WriteLine($"  {record.Date:MM/dd/yyyy}: O=${record.Open:F2} H=${record.High:F2} L=${record.Low:F2} C=${record.Close:F2} V={record.Volume:N0}");
                    }
                }
            }
        }
    }

    public class StockModelEvaluator
    {
        public StockEvaluationReport EvaluateModel(
            BayesianStockModel model,
            List<MarketFeatures> testData)
        {
            StockEvaluationReport report = new StockEvaluationReport();
            List<StockPredictionResult> predictions = new List<StockPredictionResult>();

            foreach (MarketFeatures marketData in testData)
            {
                (double predictedLow, double predictedHigh) = model.Predict(marketData);

                predictions.Add(new StockPredictionResult
                {
                    Date = marketData.Date,
                    ActualLow = marketData.MsftLow,
                    ActualHigh = marketData.MsftHigh,
                    PredictedLow = predictedLow,
                    PredictedHigh = predictedHigh,
                    LowError = Math.Abs(marketData.MsftLow - predictedLow),
                    HighError = Math.Abs(marketData.MsftHigh - predictedHigh),
                    LowPercentError = Math.Abs((marketData.MsftLow - predictedLow) / marketData.MsftLow) * 100,
                    HighPercentError = Math.Abs((marketData.MsftHigh - predictedHigh) / marketData.MsftHigh) * 100
                });
            }

            report.SampleSize = predictions.Count;
            report.LowPriceMetrics = this.CalculatePriceMetrics(predictions, true);
            report.HighPriceMetrics = this.CalculatePriceMetrics(predictions, false);
            report.DirectionalAccuracy = this.CalculateDirectionalAccuracy(predictions);
            report.ErrorDistribution = this.AnalyzeErrorDistribution(predictions);

            return report;
        }

        private PriceMetrics CalculatePriceMetrics(List<StockPredictionResult> predictions, bool isLow)
        {
            List<double> errors = isLow
                ? predictions.Select(p => p.LowError).ToList()
                : predictions.Select(p => p.HighError).ToList();

            List<double> percentErrors = isLow
                ? predictions.Select(p => p.LowPercentError).ToList()
                : predictions.Select(p => p.HighPercentError).ToList();

            return new PriceMetrics
            {
                MAE = errors.Average(),
                RMSE = Math.Sqrt(errors.Select(e => e * e).Average()),
                MAPE = percentErrors.Average(),
                MedianError = errors.OrderBy(x => x).Skip(errors.Count / 2).First(),
                MaxError = errors.Max(),
                AccuracyWithin1Percent = predictions.Count(p =>
                    (isLow ? p.LowPercentError : p.HighPercentError) <= 1.0) / (double)predictions.Count * 100,
                AccuracyWithin5Percent = predictions.Count(p =>
                    (isLow ? p.LowPercentError : p.HighPercentError) <= 5.0) / (double)predictions.Count * 100
            };
        }

        private DirectionalAccuracy CalculateDirectionalAccuracy(List<StockPredictionResult> predictions)
        {
            int correctLowDirection = 0;
            int correctHighDirection = 0;
            int correctRangeDirection = 0;

            for (int i = 1; i < predictions.Count; i++)
            {
                StockPredictionResult current = predictions[i];
                StockPredictionResult previous = predictions[i - 1];

                // Low price direction
                int actualLowDirection = Math.Sign(current.ActualLow - previous.ActualLow);
                int predictedLowDirection = Math.Sign(current.PredictedLow - previous.PredictedLow);
                if (actualLowDirection == predictedLowDirection)
                {
                    correctLowDirection++;
                }

                // High price direction
                int actualHighDirection = Math.Sign(current.ActualHigh - previous.ActualHigh);
                int predictedHighDirection = Math.Sign(current.PredictedHigh - previous.PredictedHigh);
                if (actualHighDirection == predictedHighDirection)
                {
                    correctHighDirection++;
                }

                // Range direction
                double actualRange = current.ActualHigh - current.ActualLow;
                double previousActualRange = previous.ActualHigh - previous.ActualLow;
                double predictedRange = current.PredictedHigh - current.PredictedLow;
                double previousPredictedRange = previous.PredictedHigh - previous.PredictedLow;

                int actualRangeDirection = Math.Sign(actualRange - previousActualRange);
                int predictedRangeDirection = Math.Sign(predictedRange - previousPredictedRange);
                if (actualRangeDirection == predictedRangeDirection)
                {
                    correctRangeDirection++;
                }
            }

            int totalComparisons = predictions.Count - 1;
            return new DirectionalAccuracy
            {
                LowDirectionalAccuracy = correctLowDirection / (double)totalComparisons * 100,
                HighDirectionalAccuracy = correctHighDirection / (double)totalComparisons * 100,
                RangeDirectionalAccuracy = correctRangeDirection / (double)totalComparisons * 100
            };
        }

        private StockErrorDistribution AnalyzeErrorDistribution(List<StockPredictionResult> predictions)
        {
            List<double> lowErrors = predictions.Select(p => p.LowPercentError).OrderBy(x => x).ToList();
            List<double> highErrors = predictions.Select(p => p.HighPercentError).OrderBy(x => x).ToList();

            return new StockErrorDistribution
            {
                LowErrorPercentiles = new Dictionary<string, double>
                {
                    ["P25"] = this.GetPercentile(lowErrors, 0.25),
                    ["P50"] = this.GetPercentile(lowErrors, 0.5),
                    ["P75"] = this.GetPercentile(lowErrors, 0.75),
                    ["P90"] = this.GetPercentile(lowErrors, 0.9),
                    ["P95"] = this.GetPercentile(lowErrors, 0.95)
                },
                HighErrorPercentiles = new Dictionary<string, double>
                {
                    ["P25"] = this.GetPercentile(highErrors, 0.25),
                    ["P50"] = this.GetPercentile(highErrors, 0.5),
                    ["P75"] = this.GetPercentile(highErrors, 0.75),
                    ["P90"] = this.GetPercentile(highErrors, 0.9),
                    ["P95"] = this.GetPercentile(highErrors, 0.95)
                }
            };
        }

        private double GetPercentile(List<double> sortedValues, double percentile)
        {
            int index = (int)(percentile * (sortedValues.Count - 1));
            return sortedValues[index];
        }

        public void PrintEvaluationReport(StockEvaluationReport report)
        {
            Console.WriteLine("\n" + "=".PadRight(60, '='));
            Console.WriteLine("📊 STOCK PREDICTION MODEL EVALUATION");
            Console.WriteLine("=".PadRight(60, '='));

            Console.WriteLine($"\n🎯 SAMPLE SIZE: {report.SampleSize:N0}");

            Console.WriteLine($"\n📈 LOW PRICE PREDICTIONS");
            Console.WriteLine($"   MAE: ${report.LowPriceMetrics.MAE:F3}");
            Console.WriteLine($"   RMSE: ${report.LowPriceMetrics.RMSE:F3}");
            Console.WriteLine($"   MAPE: {report.LowPriceMetrics.MAPE:F2}%");
            Console.WriteLine($"   Accuracy ≤1%: {report.LowPriceMetrics.AccuracyWithin1Percent:F1}%");
            Console.WriteLine($"   Accuracy ≤5%: {report.LowPriceMetrics.AccuracyWithin5Percent:F1}%");

            Console.WriteLine($"\n📈 HIGH PRICE PREDICTIONS");
            Console.WriteLine($"   MAE: ${report.HighPriceMetrics.MAE:F3}");
            Console.WriteLine($"   RMSE: ${report.HighPriceMetrics.RMSE:F3}");
            Console.WriteLine($"   MAPE: {report.HighPriceMetrics.MAPE:F2}%");
            Console.WriteLine($"   Accuracy ≤1%: {report.HighPriceMetrics.AccuracyWithin1Percent:F1}%");
            Console.WriteLine($"   Accuracy ≤5%: {report.HighPriceMetrics.AccuracyWithin5Percent:F1}%");

            Console.WriteLine($"\n🎯 DIRECTIONAL ACCURACY");
            Console.WriteLine($"   Low Price Direction: {report.DirectionalAccuracy.LowDirectionalAccuracy:F1}%");
            Console.WriteLine($"   High Price Direction: {report.DirectionalAccuracy.HighDirectionalAccuracy:F1}%");
            Console.WriteLine($"   Range Direction: {report.DirectionalAccuracy.RangeDirectionalAccuracy:F1}%");

            Console.WriteLine($"\n📊 ERROR DISTRIBUTION");
            Console.WriteLine($"   Low Price P50: {report.ErrorDistribution.LowErrorPercentiles["P50"]:F2}%");
            Console.WriteLine($"   Low Price P95: {report.ErrorDistribution.LowErrorPercentiles["P95"]:F2}%");
            Console.WriteLine($"   High Price P50: {report.ErrorDistribution.HighErrorPercentiles["P50"]:F2}%");
            Console.WriteLine($"   High Price P95: {report.ErrorDistribution.HighErrorPercentiles["P95"]:F2}%");
        }
    }

    public class Program
    {
        public static async Task Main(string[] args)
        {
            StockDataLoader loader = new StockDataLoader();
            MarketFeatureEngine featureEngine = new MarketFeatureEngine();
            BayesianStockModel bayesianModel = new BayesianStockModel();
            StockModelEvaluator evaluator = new StockModelEvaluator();

            try
            {
                // Load data
                Dictionary<string, List<StockData>> allStockData = await loader.LoadAllStockDataAsync();
                loader.DisplayStockSummary(allStockData);

                // Create market features
                List<MarketFeatures> marketFeatures = featureEngine.CreateMarketFeatures(allStockData);
                Console.WriteLine($"\n=== Market Features Created ===");
                Console.WriteLine($"Feature records: {marketFeatures.Count}");

                if (marketFeatures.Count > 2)
                {
                    // Split data for training and testing
                    int trainSize = (int)(marketFeatures.Count * 0.8);
                    List<MarketFeatures> trainData = marketFeatures.Take(trainSize).ToList();
                    List<MarketFeatures> testData = marketFeatures.Skip(trainSize).ToList();

                    // Train Bayesian model
                    bayesianModel.Train(trainData);
                    Console.WriteLine($"Bayesian model trained on {trainData.Count} samples");

                    // Evaluate model
                    StockEvaluationReport evaluationReport = evaluator.EvaluateModel(bayesianModel, testData);
                    evaluator.PrintEvaluationReport(evaluationReport);

                    // Make prediction on latest data
                    MarketFeatures latest = marketFeatures.Last();
                    (double predictedLow, double predictedHigh) = bayesianModel.Predict(latest);

                    Console.WriteLine($"\n=== MSFT Prediction for {latest.Date:yyyy-MM-dd} ===");
                    Console.WriteLine($"Predicted Low: ${predictedLow:F2}");
                    Console.WriteLine($"Predicted High: ${predictedHigh:F2}");
                    Console.WriteLine($"Actual Low: ${latest.MsftLow:F2}");
                    Console.WriteLine($"Actual High: ${latest.MsftHigh:F2}");
                }
            }
            catch (FileNotFoundException ex)
            {
                Console.WriteLine($"File not found: {ex.FileName}");
                Console.WriteLine("Please ensure DOW.csv, QQQ.csv, and MSFT.csv are in the application directory.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }
}