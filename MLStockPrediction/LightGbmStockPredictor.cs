namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using Microsoft.ML;
    using Microsoft.ML.Data;

    using MLStockPrediction.Models;

    public class LightGbmStockPredictor
    {
        private readonly MLContext _mlContext;
        private ITransformer _lowPriceModel;
        private ITransformer _highPriceModel;
        private ITransformer _rangePriceModel;
        private bool _isTrained = false;

        public LightGbmStockPredictor()
        {
            this._mlContext = new MLContext(seed: 0);
        }

        public void Train(List<EnhancedMarketFeatures> trainingData)
        {
            if (trainingData.Count < 10)
            {
                Console.WriteLine("⚠️  Insufficient data for LightGBM training");
                return;
            }

            Console.WriteLine($"🚀 Training LightGBM models on {trainingData.Count} samples...");

            // Convert to ML.NET format
            IDataView mlTrainingData = this._mlContext.Data.LoadFromEnumerable(
                trainingData.Select(ConvertToLightGbmInput));

            // Create feature pipeline to combine all features into a single vector
            Microsoft.ML.Transforms.ColumnConcatenatingEstimator featurePipeline = this._mlContext.Transforms.Concatenate("Features",
                // Original features
                nameof(LightGbmInput.DowReturn),
                nameof(LightGbmInput.DowVolatility),
                nameof(LightGbmInput.DowVolume),
                nameof(LightGbmInput.QqqReturn),
                nameof(LightGbmInput.QqqVolatility),
                nameof(LightGbmInput.QqqVolume),
                nameof(LightGbmInput.MsftReturn),
                nameof(LightGbmInput.MsftVolatility),
                nameof(LightGbmInput.MsftVolume),

                // Technical features
                nameof(LightGbmInput.DowSMA5),
                nameof(LightGbmInput.DowSMA10),
                nameof(LightGbmInput.DowSMA20),
                nameof(LightGbmInput.QqqSMA5),
                nameof(LightGbmInput.QqqSMA10),
                nameof(LightGbmInput.QqqSMA20),
                nameof(LightGbmInput.MsftSMA5),
                nameof(LightGbmInput.MsftSMA10),
                nameof(LightGbmInput.MsftSMA20),

                nameof(LightGbmInput.DowEMAR5),
                nameof(LightGbmInput.DowEMAR10),
                nameof(LightGbmInput.DowEMAR20),
                nameof(LightGbmInput.QqqEMAR5),
                nameof(LightGbmInput.QqqEMAR10),
                nameof(LightGbmInput.QqqEMAR20),
                nameof(LightGbmInput.MsftEMAR5),
                nameof(LightGbmInput.MsftEMAR10),
                nameof(LightGbmInput.MsftEMAR20),

                nameof(LightGbmInput.DowPricePosition),
                nameof(LightGbmInput.QqqPricePosition),
                nameof(LightGbmInput.MsftPricePosition),

                nameof(LightGbmInput.DowROC5),
                nameof(LightGbmInput.DowROC10),
                nameof(LightGbmInput.QqqROC5),
                nameof(LightGbmInput.QqqROC10),
                nameof(LightGbmInput.MsftROC5),
                nameof(LightGbmInput.MsftROC10),

                nameof(LightGbmInput.DowVolatility5),
                nameof(LightGbmInput.DowVolatility10),
                nameof(LightGbmInput.QqqVolatility5),
                nameof(LightGbmInput.QqqVolatility10),
                nameof(LightGbmInput.MsftVolatility5),
                nameof(LightGbmInput.MsftVolatility10),

                // Temporal features
                nameof(LightGbmInput.IsMondayEffect),
                nameof(LightGbmInput.IsTuesdayEffect),
                nameof(LightGbmInput.IsWednesdayEffect),
                nameof(LightGbmInput.IsThursdayEffect),
                nameof(LightGbmInput.IsFridayEffect),
                nameof(LightGbmInput.IsOptionsExpirationWeek),
                nameof(LightGbmInput.IsQuarterStart),
                nameof(LightGbmInput.IsQuarterEnd),
                nameof(LightGbmInput.QuarterProgress),
                nameof(LightGbmInput.YearProgress)
            );

            // Train individual models, passing the feature pipeline to them
            this.TrainLowPriceModel(mlTrainingData, featurePipeline);
            this.TrainHighPriceModel(mlTrainingData, featurePipeline);
            this.TrainRangePriceModel(mlTrainingData, featurePipeline);

            this._isTrained = true;
            Console.WriteLine("✅ LightGBM models training completed");
        }

        public (double Low, double High, double Range) Predict(EnhancedMarketFeatures marketData)
        {
            if (!this._isTrained)
            {
                throw new InvalidOperationException("LightGBM models must be trained first");
            }

            LightGbmInput input = ConvertToLightGbmInput(marketData);

            // The feature transformation is now part of the model's pipeline,
            // so we can create the prediction engine directly.
            PredictionEngine<LightGbmInput, LightGbmLowOutput> lowPricePredictionEngine = this._mlContext.Model.CreatePredictionEngine<LightGbmInput, LightGbmLowOutput>(this._lowPriceModel);
            PredictionEngine<LightGbmInput, LightGbmHighOutput> highPricePredictionEngine = this._mlContext.Model.CreatePredictionEngine<LightGbmInput, LightGbmHighOutput>(this._highPriceModel);
            PredictionEngine<LightGbmInput, LightGbmRangeOutput> rangePricePredictionEngine = this._mlContext.Model.CreatePredictionEngine<LightGbmInput, LightGbmRangeOutput>(this._rangePriceModel);

            // Get predictions from all models
            LightGbmLowOutput lowPrediction = lowPricePredictionEngine.Predict(input);
            LightGbmHighOutput highPrediction = highPricePredictionEngine.Predict(input);
            LightGbmRangeOutput rangePrediction = rangePricePredictionEngine.Predict(input);

            return (lowPrediction.PredictedLowPrice, highPrediction.PredictedHighPrice, rangePrediction.PredictedRange);
        }

        private void TrainLowPriceModel(IDataView trainingData, IEstimator<ITransformer> featurePipeline)
        {
            Console.WriteLine("🎯 Training LightGBM Low Price Model...");

            // Start with the feature pipeline and append the trainer
            EstimatorChain<RegressionPredictionTransformer<Microsoft.ML.Trainers.LightGbm.LightGbmRegressionModelParameters>> pipeline = featurePipeline
                .Append(this._mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(LightGbmInput.LowPrice)))
                .Append(this._mlContext.Regression.Trainers.LightGbm(new Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options
                {
                    LabelColumnName = "Label",
                    FeatureColumnName = "Features",
                    NumberOfLeaves = 64,
                    MinimumExampleCountPerLeaf = 10,
                    LearningRate = 0.04,
                    NumberOfIterations = 200,
                    HandleMissingValue = true,
                    UseCategoricalSplit = true,
                    CategoricalSmoothing = 10.0,
                    EarlyStoppingRound = 20
                }));

            this._lowPriceModel = pipeline.Fit(trainingData);
        }

        private void TrainHighPriceModel(IDataView trainingData, IEstimator<ITransformer> featurePipeline)
        {
            Console.WriteLine("🎯 Training LightGBM High Price Model...");

            // Start with the feature pipeline and append the trainer
            EstimatorChain<RegressionPredictionTransformer<Microsoft.ML.Trainers.LightGbm.LightGbmRegressionModelParameters>> pipeline = featurePipeline
                .Append(this._mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(LightGbmInput.HighPrice)))
                .Append(this._mlContext.Regression.Trainers.LightGbm(new Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options
                {
                    LabelColumnName = "Label",
                    FeatureColumnName = "Features",
                    NumberOfLeaves = 64,
                    MinimumExampleCountPerLeaf = 8,
                    LearningRate = 0.04,
                    NumberOfIterations = 200,
                    HandleMissingValue = true,
                    UseCategoricalSplit = true,
                    CategoricalSmoothing = 10.0,
                    EarlyStoppingRound = 20
                }));

            this._highPriceModel = pipeline.Fit(trainingData);
        }

        private void TrainRangePriceModel(IDataView trainingData, IEstimator<ITransformer> featurePipeline)
        {
            Console.WriteLine("🎯 Training LightGBM Range Model...");

            // Start with the feature pipeline and append the trainer
            EstimatorChain<RegressionPredictionTransformer<Microsoft.ML.Trainers.LightGbm.LightGbmRegressionModelParameters>> pipeline = featurePipeline
                .Append(this._mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(LightGbmInput.PriceRange)))
                .Append(this._mlContext.Regression.Trainers.LightGbm(new Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options
                {
                    LabelColumnName = "Label",
                    FeatureColumnName = "Features",
                    NumberOfLeaves = 32,
                    MinimumExampleCountPerLeaf = 15,
                    LearningRate = 0.1,
                    NumberOfIterations = 150,
                    HandleMissingValue = true,
                    UseCategoricalSplit = false,
                    CategoricalSmoothing = 20.0,
                    EarlyStoppingRound = 15
                }));

            this._rangePriceModel = pipeline.Fit(trainingData);
        }

        private static LightGbmInput ConvertToLightGbmInput(EnhancedMarketFeatures features)
        {
            return new LightGbmInput
            {
                // ... (this method remains unchanged)
                // Original features
                DowReturn = (float)features.DowReturn,
                DowVolatility = (float)features.DowVolatility,
                DowVolume = (float)features.DowVolume,
                QqqReturn = (float)features.QqqReturn,
                QqqVolatility = (float)features.QqqVolatility,
                QqqVolume = (float)features.QqqVolume,
                MsftReturn = (float)features.MsftReturn,
                MsftVolatility = (float)features.MsftVolatility,
                MsftVolume = (float)features.MsftVolume,

                // Technical features
                DowSMA5 = (float)features.DowSMA5,
                DowSMA10 = (float)features.DowSMA10,
                DowSMA20 = (float)features.DowSMA20,
                QqqSMA5 = (float)features.QqqSMA5,
                QqqSMA10 = (float)features.QqqSMA10,
                QqqSMA20 = (float)features.QqqSMA20,
                MsftSMA5 = (float)features.MsftSMA5,
                MsftSMA10 = (float)features.MsftSMA10,
                MsftSMA20 = (float)features.MsftSMA20,

                DowEMAR5 = (float)features.DowEMAR5,
                DowEMAR10 = (float)features.DowEMAR10,
                DowEMAR20 = (float)features.DowEMAR20,
                QqqEMAR5 = (float)features.QqqEMAR5,
                QqqEMAR10 = (float)features.QqqEMAR10,
                QqqEMAR20 = (float)features.QqqEMAR20,
                MsftEMAR5 = (float)features.MsftEMAR5,
                MsftEMAR10 = (float)features.MsftEMAR10,
                MsftEMAR20 = (float)features.MsftEMAR20,

                DowPricePosition = (float)features.DowPricePosition,
                QqqPricePosition = (float)features.QqqPricePosition,
                MsftPricePosition = (float)features.MsftPricePosition,

                DowROC5 = (float)features.DowROC5,
                DowROC10 = (float)features.DowROC10,
                QqqROC5 = (float)features.QqqROC5,
                QqqROC10 = (float)features.QqqROC10,
                MsftROC5 = (float)features.MsftROC5,
                MsftROC10 = (float)features.MsftROC10,

                DowVolatility5 = (float)features.DowVolatility5,
                DowVolatility10 = (float)features.DowVolatility10,
                QqqVolatility5 = (float)features.QqqVolatility5,
                QqqVolatility10 = (float)features.QqqVolatility10,
                MsftVolatility5 = (float)features.MsftVolatility5,
                MsftVolatility10 = (float)features.MsftVolatility10,

                // Temporal features
                IsMondayEffect = (float)features.IsMondayEffect,
                IsTuesdayEffect = (float)features.IsTuesdayEffect,
                IsWednesdayEffect = (float)features.IsWednesdayEffect,
                IsThursdayEffect = (float)features.IsThursdayEffect,
                IsFridayEffect = (float)features.IsFridayEffect,

                IsOptionsExpirationWeek = (float)features.IsOptionsExpirationWeek,
                IsQuarterStart = (float)features.IsQuarterStart,
                IsQuarterEnd = (float)features.IsQuarterEnd,
                QuarterProgress = (float)features.QuarterProgress,
                YearProgress = (float)features.YearProgress,

                // Target variables
                LowPrice = (float)features.MsftLow,
                HighPrice = (float)features.MsftHigh,
                PriceRange = (float)(features.MsftHigh - features.MsftLow)
            };
        }
    }
}