import calendar
import sys
from datetime import datetime


def display_current_month():
    """Display the current month's calendar."""
    now = datetime.now()
    cal = calendar.TextCalendar()
    print(f"\n{'=' * 28}")
    print(f"   {calendar.month_name[now.month]} {now.year}")
    print(f"{'=' * 28}")
    cal.prmonth(now.year, now.month)


def display_current_year():
    """Display the entire current year's calendar."""
    now = datetime.now()
    print(f"\n{'=' * 40}")
    print(f"          {now.year}")
    print(f"{'=' * 40}")
    cal = calendar.TextCalendar()
    print(cal.formatyear(now.year))


def format_month_lines(year, month):
    """Return the calendar for a given month as a list of lines."""
    cal = calendar.TextCalendar()
    lines = cal.formatmonth(year, month).splitlines()
    return lines


def display_context_months(offset):
    """Display months around the current month, max 3 per row."""
    now = datetime.now()
    current_month = now.month
    current_year = now.year

    start_month = current_month - offset
    start_year = current_year
    end_month = current_month + offset
    end_year = current_year

    # Adjust start year/month
    while start_month < 1:
        start_month += 12
        start_year -= 1

    # Adjust end year/month
    while end_month > 12:
        end_month -= 12
        end_year += 1

    # Collect all months
    months_data = []
    year = start_year
    month = start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        months_data.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    print(f"\n{'=' * 28}")
    print(f"  Context: +/-{offset} month(s)")
    print(f"{'=' * 28}")

    # Display months in rows of 3
    for i in range(0, len(months_data), 3):
        row = months_data[i:i + 3]
        # Get formatted lines for each month in the row
        month_lines = [format_month_lines(y, m) for y, m in row]

        # Determine max height for this row
        max_lines = max(len(lines) for lines in month_lines)

        # Pad each month's lines to the same height
        for lines in month_lines:
            while len(lines) < max_lines:
                lines.append(" " * len(lines[0]) if lines else "")

        # Print the row side by side
        for line_idx in range(max_lines):
            parts = []
            for lines in month_lines:
                parts.append(lines[line_idx])
            print("   ".join(parts))


def print_usage():
    print("Usage: python my_calendar.py <option>")
    print()
    print("Options:")
    print("  month            Display the current month")
    print("  year             Display the entire current year")
    print("  1                Display current month +/-1 month")
    print("  3                Display current month +/-3 months")
    print("  -h, --help       Show this help message")


def main():
    if len(sys.argv) == 1:
        display_context_months(1)
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg in ("-h", "--help"):
            print_usage()
        elif arg == "month":
            display_current_month()
        elif arg == "year":
            display_current_year()
        elif arg == "1":
            display_context_months(1)
        elif arg == "3":
            display_context_months(3)
        else:
            print(f"Unknown option: {arg}")
            print_usage()
    else:
        print_usage()


if __name__ == "__main__":
    main()