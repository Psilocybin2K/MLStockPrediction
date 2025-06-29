namespace MLStockPrediction
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MLStockPrediction.Models;

    public static class TemporalFeatureCalculator
    {
        private static readonly HashSet<DateTime> MarketHolidays2025 = new()
        {
            new DateTime(2025, 1, 1),   // New Year's Day
            new DateTime(2025, 1, 20),  // MLK Day
            new DateTime(2025, 2, 17),  // Presidents' Day
            new DateTime(2025, 4, 18),  // Good Friday
            new DateTime(2025, 5, 26),  // Memorial Day
            new DateTime(2025, 7, 4),   // Independence Day
            new DateTime(2025, 9, 1),   // Labor Day
            new DateTime(2025, 11, 27), // Thanksgiving
            new DateTime(2025, 12, 25)  // Christmas
        };

        private static readonly Dictionary<int, DateTime> EarningsSeasonStarts = new()
        {
            { 1, new DateTime(2025, 1, 6) },   // Q4 earnings
            { 4, new DateTime(2025, 4, 7) },   // Q1 earnings
            { 7, new DateTime(2025, 7, 7) },   // Q2 earnings
            { 10, new DateTime(2025, 10, 6) }  // Q3 earnings
        };

        public static void CalculateTemporalFeatures(EnhancedMarketFeatures feature)
        {
            DateTime date = feature.Date;

            // Day of Week Effects
            feature.IsMondayEffect = date.DayOfWeek == DayOfWeek.Monday ? 1.0 : 0.0;
            feature.IsTuesdayEffect = date.DayOfWeek == DayOfWeek.Tuesday ? 1.0 : 0.0;
            feature.IsWednesdayEffect = date.DayOfWeek == DayOfWeek.Wednesday ? 1.0 : 0.0;
            feature.IsThursdayEffect = date.DayOfWeek == DayOfWeek.Thursday ? 1.0 : 0.0;
            feature.IsFridayEffect = date.DayOfWeek == DayOfWeek.Friday ? 1.0 : 0.0;

            // Week of Month Patterns
            int weekOfMonth = GetWeekOfMonth(date);
            feature.IsFirstWeekOfMonth = weekOfMonth == 1 ? 1.0 : 0.0;
            feature.IsSecondWeekOfMonth = weekOfMonth == 2 ? 1.0 : 0.0;
            feature.IsThirdWeekOfMonth = weekOfMonth == 3 ? 1.0 : 0.0;
            feature.IsFourthWeekOfMonth = weekOfMonth >= 4 ? 1.0 : 0.0;
            feature.IsOptionsExpirationWeek = IsOptionsExpirationWeek(date) ? 1.0 : 0.0;

            // Month Effects
            feature.IsJanuaryEffect = date.Month == 1 ? 1.0 : 0.0;
            feature.IsQuarterStart = IsQuarterStart(date) ? 1.0 : 0.0;
            feature.IsQuarterEnd = IsQuarterEnd(date) ? 1.0 : 0.0;
            feature.IsYearEnd = date.Month == 12 ? 1.0 : 0.0;

            // Holiday Proximity
            (int daysToBefore, int daysFromAfter) = GetHolidayProximity(date);
            feature.DaysToMarketHoliday = Math.Min(daysToBefore, 10); // Cap at 10 days
            feature.DaysFromMarketHoliday = Math.Min(daysFromAfter, 10);

            // Quarter Progress
            DateTime quarterStart = GetQuarterStart(date);
            DateTime quarterEnd = GetQuarterEnd(date);
            int daysIntoQuarter = (date - quarterStart).Days;
            int totalQuarterDays = (quarterEnd - quarterStart).Days;

            feature.DaysIntoQuarter = daysIntoQuarter;
            feature.DaysUntilQuarterEnd = (quarterEnd - date).Days;
            feature.QuarterProgress = (double)daysIntoQuarter / totalQuarterDays;

            // Earnings Season
            (bool isEarnings, int daysToEarnings, int daysFromEarnings) = GetEarningsProximity(date);
            feature.IsEarningsSeason = isEarnings ? 1.0 : 0.0;
            feature.DaysToEarningsWeek = Math.Min(daysToEarnings, 30);
            feature.DaysFromEarningsWeek = Math.Min(daysFromEarnings, 30);

            // Year Progress
            DateTime yearStart = new DateTime(date.Year, 1, 1);
            DateTime yearEnd = new DateTime(date.Year, 12, 31);
            feature.YearProgress = (double)(date - yearStart).Days / (yearEnd - yearStart).Days;

            // Month Progress
            DateTime monthStart = new DateTime(date.Year, date.Month, 1);
            DateTime monthEnd = monthStart.AddMonths(1).AddDays(-1);
            feature.MonthProgress = (double)(date - monthStart).Days / (monthEnd - monthStart).Days;
        }

        private static int GetWeekOfMonth(DateTime date)
        {
            DateTime firstOfMonth = new DateTime(date.Year, date.Month, 1);
            int firstWeekday = (int)firstOfMonth.DayOfWeek;
            return (date.Day + firstWeekday - 1) / 7 + 1;
        }

        private static bool IsOptionsExpirationWeek(DateTime date)
        {
            // Third Friday of each month
            DateTime thirdFriday = GetNthWeekdayOfMonth(date.Year, date.Month, DayOfWeek.Friday, 3);
            DateTime weekStart = thirdFriday.AddDays(-(int)thirdFriday.DayOfWeek + 1);
            DateTime weekEnd = weekStart.AddDays(6);

            return date >= weekStart && date <= weekEnd;
        }

        private static DateTime GetNthWeekdayOfMonth(int year, int month, DayOfWeek dayOfWeek, int n)
        {
            DateTime firstOfMonth = new DateTime(year, month, 1);
            int daysToAdd = ((int)dayOfWeek - (int)firstOfMonth.DayOfWeek + 7) % 7;
            DateTime firstOccurrence = firstOfMonth.AddDays(daysToAdd);
            return firstOccurrence.AddDays((n - 1) * 7);
        }

        private static bool IsQuarterStart(DateTime date)
        {
            return (date.Month == 1 || date.Month == 4 || date.Month == 7 || date.Month == 10)
                   && date.Day <= 7;
        }

        private static bool IsQuarterEnd(DateTime date)
        {
            return (date.Month == 3 || date.Month == 6 || date.Month == 9 || date.Month == 12)
                   && date.Day >= 25;
        }

        private static DateTime GetQuarterStart(DateTime date)
        {
            int quarter = ((date.Month - 1) / 3) + 1;
            int startMonth = (quarter - 1) * 3 + 1;
            return new DateTime(date.Year, startMonth, 1);
        }

        private static DateTime GetQuarterEnd(DateTime date)
        {
            int quarter = ((date.Month - 1) / 3) + 1;
            int endMonth = quarter * 3;
            return new DateTime(date.Year, endMonth, DateTime.DaysInMonth(date.Year, endMonth));
        }

        private static (int daysToBefore, int daysFromAfter) GetHolidayProximity(DateTime date)
        {
            DateTime closestBefore = MarketHolidays2025.Where(h => h < date).DefaultIfEmpty(DateTime.MinValue).Max();
            DateTime closestAfter = MarketHolidays2025.Where(h => h > date).DefaultIfEmpty(DateTime.MaxValue).Min();

            int daysToBefore = closestBefore == DateTime.MinValue ? 365 : (date - closestBefore).Days;
            int daysFromAfter = closestAfter == DateTime.MaxValue ? 365 : (closestAfter - date).Days;

            return (daysFromAfter, daysToBefore);
        }

        private static (bool isEarnings, int daysTo, int daysFrom) GetEarningsProximity(DateTime date)
        {
            // Find current quarter's earnings season
            int currentQuarter = ((date.Month - 1) / 3) + 1;

            // Earnings seasons are typically 1 month after quarter end
            DateTime earningsStart = currentQuarter switch
            {
                1 => EarningsSeasonStarts[1], // Q4 previous year earnings in January
                2 => EarningsSeasonStarts[4], // Q1 earnings in April
                3 => EarningsSeasonStarts[7], // Q2 earnings in July  
                4 => EarningsSeasonStarts[10], // Q3 earnings in October
                _ => DateTime.MinValue
            };

            DateTime earningsEnd = earningsStart.AddDays(21); // ~3 weeks
            bool isInSeason = date >= earningsStart && date <= earningsEnd;

            // Find next earnings season
            DateTime nextEarnings = EarningsSeasonStarts.Values
                .Where(e => e > date)
                .DefaultIfEmpty(EarningsSeasonStarts[1].AddYears(1))
                .Min();

            // Find previous earnings season  
            DateTime prevEarnings = EarningsSeasonStarts.Values
                .Where(e => e.AddDays(21) < date)
                .DefaultIfEmpty(EarningsSeasonStarts[10].AddYears(-1))
                .Max();

            int daysTo = (nextEarnings - date).Days;
            int daysFrom = (date - prevEarnings.AddDays(21)).Days;

            return (isInSeason, Math.Max(0, daysTo), Math.Max(0, daysFrom));
        }
    }
}