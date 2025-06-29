namespace MLStockPrediction.Converters
{
    using System.Globalization;

    using CsvHelper;

    public class DecimalConverter : CsvHelper.TypeConversion.DecimalConverter
    {
        public override object ConvertFromString(string text, IReaderRow row, CsvHelper.Configuration.MemberMapData memberMapData)
        {
            if (string.IsNullOrEmpty(text))
            {
                return 0m;
            }

            string cleanText = text.Replace("$", "").Replace(",", "");
            return decimal.Parse(cleanText, CultureInfo.InvariantCulture);
        }
    }
}