import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def load_from_csv(nifty_csv_path, vix_csv_path):
    """
    Load NIFTY and VIX data from CSV files
    
    Args:
        nifty_csv_path: Path to NIFTY CSV file
        vix_csv_path: Path to VIX CSV file
        
    Returns:
        Tuple of (nifty_df, vix_df)
    """
    try:
        nifty_df = pd.read_csv(nifty_csv_path, index_col=0, parse_dates=True)
        vix_df = pd.read_csv(vix_csv_path, index_col=0, parse_dates=True)
        
        # Ensure DatetimeIndex
        if not isinstance(nifty_df.index, pd.DatetimeIndex):
            nifty_df.index = pd.to_datetime(nifty_df.index)
            
        if not isinstance(vix_df.index, pd.DatetimeIndex):
            vix_df.index = pd.to_datetime(vix_df.index)
            
        print(f"Loaded {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records from CSV")
        return nifty_df, vix_df
    
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        return None, None

def fetch_new_data(start_date=None, end_date=None, days=30):
    """
    Fetch fresh NIFTY and VIX data from Yahoo Finance
    
    Args:
        start_date: Start date for data fetching
        end_date: End date for data fetching
        days: Number of days to fetch if start_date not specified
        
    Returns:
        Tuple of (nifty_df, vix_df)
    """
    try:
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now(ZoneInfo("Asia/Kolkata"))
            
        if start_date is None:
            start_date = end_date - timedelta(days=days)
        
        print(f"Fetching data from {start_date.date()} to {end_date.date()}")
        
        # Get NIFTY data
        nifty_df = get_chunked_data("^NSEI", start_date, end_date)
        
        # Get India VIX data
        vix_df = get_chunked_data("^INDIAVIX", start_date, end_date)
        
        print(f"Fetched {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records")
        return nifty_df, vix_df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None

def get_chunked_data(ticker_symbol, start_date, end_date, chunk_days=5):
    """
    Get data in chunks to avoid missing data and rate limits
    
    Args:
        ticker_symbol: Yahoo Finance ticker symbol
        start_date: Start date
        end_date: End date
        chunk_days: Number of days per chunk
        
    Returns:
        DataFrame with combined data
    """
    chunks = []
    current_start = start_date
    
    while current_start < end_date:
        # Use smaller chunks to avoid data gaps
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
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

def save_to_csv(nifty_df, vix_df, data_dir='data'):
    """
    Save data to CSV files
    
    Args:
        nifty_df: NIFTY DataFrame
        vix_df: VIX DataFrame
        data_dir: Directory to save data
        
    Returns:
        Tuple of (nifty_csv_path, vix_csv_path)
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate filenames with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nifty_csv_path = os.path.join(data_dir, f'nifty_data_{timestamp}.csv')
    vix_csv_path = os.path.join(data_dir, f'vix_data_{timestamp}.csv')
    
    # Save data
    nifty_df.to_csv(nifty_csv_path)
    vix_df.to_csv(vix_csv_path)
    
    print(f"Data saved to {nifty_csv_path} and {vix_csv_path}")
    return nifty_csv_path, vix_csv_path

def get_latest_data(data_dir='data', load_from_disk=True, fetch_days=30):
    """
    Get the latest data, either from disk or by fetching from Yahoo Finance
    
    Args:
        data_dir: Directory with data files
        load_from_disk: Whether to try loading from disk first
        fetch_days: Number of days to fetch if downloading new data
        
    Returns:
        Tuple of (nifty_df, vix_df)
    """
    nifty_df, vix_df = None, None
    
    if load_from_disk:
        # Find latest CSV files in data directory
        try:
            nifty_files = [f for f in os.listdir(data_dir) if f.startswith('nifty_data_') and f.endswith('.csv')]
            vix_files = [f for f in os.listdir(data_dir) if f.startswith('vix_data_') and f.endswith('.csv')]
            
            if nifty_files and vix_files:
                latest_nifty = sorted(nifty_files)[-1]
                latest_vix = sorted(vix_files)[-1]
                
                nifty_path = os.path.join(data_dir, latest_nifty)
                vix_path = os.path.join(data_dir, latest_vix)
                
                nifty_df, vix_df = load_from_csv(nifty_path, vix_path)
        except Exception as e:
            print(f"Error loading data from disk: {e}")
    
    # If loading from disk failed or was skipped, fetch new data
    if nifty_df is None or vix_df is None:
        nifty_df, vix_df = fetch_new_data(days=fetch_days)
        
        # Save new data
        if nifty_df is not None and vix_df is not None:
            save_to_csv(nifty_df, vix_df, data_dir)
    
    return nifty_df, vix_df 