from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from zoneinfo import ZoneInfo

# Set the end date to today with the correct timezone (IST for NIFTY)
end_date = datetime.now(ZoneInfo("Asia/Kolkata"))
# Calculate start date as exactly 30 days before end_date
start_date = end_date - timedelta(days=24)

def get_chunked_data(ticker_symbol):
    chunks = []
    current_start = start_date
    
    while current_start < end_date:
        # Use smaller chunks (5 days) to avoid data gaps
        current_end = min(current_start + timedelta(days=5), end_date)
        try:
            ticker = yf.Ticker(ticker_symbol)
            chunk = ticker.history(
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d"),
                interval="1m",
                prepost=True  # Include pre and post market data
            )
            if not chunk.empty:
                # Convert timezone to IST for both indices
                chunk.index = chunk.index.tz_convert('Asia/Kolkata')
                chunks.append(chunk)
        except Exception as e:
            print(f"Error fetching data for {ticker_symbol} from {current_start} to {current_end}: {e}")
        
        current_start = current_end
    
    if not chunks:
        raise ValueError(f"No data retrieved for {ticker_symbol}")
        
    return pd.concat(chunks).sort_index().drop_duplicates()

try:
    # Get NIFTY data
    print(f"Fetching NIFTY data from {start_date.date()} to {end_date.date()}")
    nifty_df = get_chunked_data("^NSEI")
    nifty_df.to_csv('nifty_1min_data.csv')
    
    # Get India VIX data
    print(f"Fetching India VIX data from {start_date.date()} to {end_date.date()}")
    vix_df = get_chunked_data("^INDIAVIX")
    vix_df.to_csv('vix_1min_data.csv')
    
    print(f"Data saved successfully")
    print(f"NIFTY records: {len(nifty_df)}")
    print(f"India VIX records: {len(vix_df)}")

except Exception as e:
    print(f"An error occurred: {e}")
