namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using Microsoft.ML;

    using MLStockPrediction.Models;

    public class EnsembleStockModel
    {
        private readonly BayesianStockModel _bayesianModel;
        private readonly LightGbmStockPredictor _lightGbmModel;
        private ITransformer _metaLowModel;
        private ITransformer _metaHighModel;
        private readonly MLContext _mlContext;
        private bool _isTrained = false;

        public EnsembleStockModel()
        {
            this._bayesianModel = new BayesianStockModel();
            this._lightGbmModel = new LightGbmStockPredictor();
            this._mlContext = new MLContext(seed: 0);
        }

        public void Train(List<EnhancedMarketFeatures> trainingData)
        {
            if (trainingData.Count < 50)
            {
                Console.WriteLine("⚠️  Insufficient data for k-fold stacking ensemble training");
                return;
            }

            Console.WriteLine("🚀 Training Stacking Ensemble Model with k-Fold Cross-Validation...");

            // 1. Generate meta-features using k-fold cross-validation
            int kFolds = 5;
            List<StackingMetaInput> metaFeatures = this.GenerateMetaFeaturesWithKFold(trainingData, kFolds);

            if (metaFeatures.Count == 0)
            {
                Console.WriteLine("❌ Failed to generate meta-features. Aborting training.");
                return;
            }

            // 2. Train the meta-models on the generated out-of-sample predictions
            IDataView metaTrainingData = this._mlContext.Data.LoadFromEnumerable(metaFeatures);
            this.TrainMetaModels(metaTrainingData);

            // 3. Retrain the base models on the *entire* training dataset so they are as accurate as possible for future predictions
            Console.WriteLine("📊 Retraining base models on full training data...");
            this._bayesianModel.Train(trainingData);
            this._lightGbmModel.Train(trainingData);

            this._isTrained = true;
            Console.WriteLine("✅ Stacking ensemble model training completed");
        }

        private List<StackingMetaInput> GenerateMetaFeaturesWithKFold(List<EnhancedMarketFeatures> data, int k)
        {
            Console.WriteLine($"🧠 Generating meta-features using {k}-fold cross-validation...");
            List<StackingMetaInput> outOfSamplePredictions = new List<StackingMetaInput>();
            int foldSize = data.Count / k;

            for (int i = 0; i < k; i++)
            {
                Console.WriteLine($"-- Fold {i + 1}/{k} --");

                // Define the training and validation sets for this fold
                List<EnhancedMarketFeatures> validationFold = data.Skip(i * foldSize).Take(foldSize).ToList();
                List<EnhancedMarketFeatures> trainingFolds = data.Take(i * foldSize).Concat(data.Skip((i + 1) * foldSize)).ToList();

                if (validationFold.Count == 0 || trainingFolds.Count == 0)
                {
                    continue;
                }

                // Train temporary base models on the training folds
                BayesianStockModel tempBayesianModel = new BayesianStockModel();
                LightGbmStockPredictor tempLightGbmModel = new LightGbmStockPredictor();

                tempBayesianModel.Train(trainingFolds);
                tempLightGbmModel.Train(trainingFolds);

                // Make predictions on the validation fold
                foreach (EnhancedMarketFeatures sample in validationFold)
                {
                    (double bayesianLow, double bayesianHigh) = tempBayesianModel.Predict(sample);
                    (double lightGbmLow, double lightGbmHigh, _) = tempLightGbmModel.Predict(sample);

                    outOfSamplePredictions.Add(new StackingMetaInput
                    {
                        BayesianLow = (float)bayesianLow,
                        BayesianHigh = (float)bayesianHigh,
                        LightGbmLow = (float)lightGbmLow,
                        LightGbmHigh = (float)lightGbmHigh,
                        ActualLow = (float)sample.MsftLow,
                        ActualHigh = (float)sample.MsftHigh
                    });
                }
            }

            return outOfSamplePredictions;
        }

        private void TrainMetaModels(IDataView metaTrainingData)
        {
            Console.WriteLine("🧠 Training Meta-Models on out-of-sample data...");

            Microsoft.ML.Transforms.ColumnConcatenatingEstimator featurePipeline = this._mlContext.Transforms.Concatenate("Features",
                nameof(StackingMetaInput.BayesianLow),
                nameof(StackingMetaInput.BayesianHigh),
                nameof(StackingMetaInput.LightGbmLow),
                nameof(StackingMetaInput.LightGbmHigh));

            // Meta-model for the Low price (using a simpler model like SDCA)
            Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.LinearRegressionModelParameters>> lowPricePipeline = featurePipeline
                .Append(this._mlContext.Transforms.CopyColumns("Label", nameof(StackingMetaInput.ActualLow)))
                .Append(this._mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));

            this._metaLowModel = lowPricePipeline.Fit(metaTrainingData);

            // Meta-model for the High price
            Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.LinearRegressionModelParameters>> highPricePipeline = featurePipeline
                .Append(this._mlContext.Transforms.CopyColumns("Label", nameof(StackingMetaInput.ActualHigh)))
                .Append(this._mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));

            this._metaHighModel = highPricePipeline.Fit(metaTrainingData);
        }

        public EnsemblePredictionResult Predict(EnhancedMarketFeatures marketData)
        {
            if (!this._isTrained)
            {
                throw new InvalidOperationException("Ensemble model must be trained first");
            }

            // Get predictions from base models (which were retrained on full data)
            (double bayesianLow, double bayesianHigh) = this._bayesianModel.Predict(marketData);
            (double lightGbmLow, double lightGbmHigh, double lightGbmRange) = this._lightGbmModel.Predict(marketData);

            // Create input for the meta-model
            StackingMetaInput metaInput = new StackingMetaInput
            {
                BayesianLow = (float)bayesianLow,
                BayesianHigh = (float)bayesianHigh,
                LightGbmLow = (float)lightGbmLow,
                LightGbmHigh = (float)lightGbmHigh
            };

            // Predict with the meta-models
            PredictionEngine<StackingMetaInput, LightGbmLowOutput> lowPredictionEngine = this._mlContext.Model.CreatePredictionEngine<StackingMetaInput, LightGbmLowOutput>(this._metaLowModel);
            PredictionEngine<StackingMetaInput, LightGbmHighOutput> highPredictionEngine = this._mlContext.Model.CreatePredictionEngine<StackingMetaInput, LightGbmHighOutput>(this._metaHighModel);

            float finalLow = lowPredictionEngine.Predict(metaInput).PredictedLowPrice;
            float finalHigh = highPredictionEngine.Predict(metaInput).PredictedHighPrice;

            // Ensure high >= low
            if (finalHigh < finalLow)
            {
                (finalLow, finalHigh) = (finalHigh, finalLow);
            }

            return new EnsemblePredictionResult
            {
                Date = marketData.Date,
                BayesianLow = bayesianLow,
                BayesianHigh = bayesianHigh,
                LightGbmLow = lightGbmLow,
                LightGbmHigh = lightGbmHigh,
                FinalLow = finalLow,
                FinalHigh = finalHigh,
            };
        }

        public void EnableCalibration(bool enable = true)
        {
            this._bayesianModel.EnableCalibration(enable);
        }
    }
}