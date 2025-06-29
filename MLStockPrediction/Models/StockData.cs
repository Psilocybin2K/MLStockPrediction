namespace MLStockPrediction.Models
{
    using System;

    using CsvHelper.Configuration.Attributes;

    public class StockData
    {
        public DateTime Date { get; set; }

        [Name("Close/Last")]
        public decimal Close { get; set; }

        public long Volume { get; set; }
        public decimal Open { get; set; }
        public decimal High { get; set; }
        public decimal Low { get; set; }
    }
}