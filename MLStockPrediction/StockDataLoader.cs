namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;

    using CsvHelper;

    using MLStockPrediction.Converters;
    using MLStockPrediction.Models;

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
}