import pandas as pd
import yfinance as yf
import os
import glob
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
            
        # Convert to timezone-naive datetime for consistency
        if nifty_df.index.tz is not None:
            nifty_df = nifty_df.tz_localize(None)
        if vix_df.index.tz is not None:
            vix_df = vix_df.tz_localize(None)
            
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
                # Convert to naive datetime (without timezone) for consistency
                if chunk.index.tz is not None:
                    chunk.index = chunk.index.tz_convert('Asia/Kolkata').tz_localize(None)
                chunks.append(chunk)
        except Exception as e:
            print(f"Error fetching data for {ticker_symbol} from {current_start} to {current_end}: {e}")
        
        current_start = current_end
    
    if not chunks:
        raise ValueError(f"No data retrieved for {ticker_symbol}")
        
    result = pd.concat(chunks).sort_index().drop_duplicates()
    return result

def save_to_csv(nifty_df, vix_df, data_dir='data', consolidated=True):
    """
    Save data to CSV files
    
    Args:
        nifty_df: NIFTY DataFrame
        vix_df: VIX DataFrame
        data_dir: Directory to save data
        consolidated: Whether to save as consolidated files or with timestamp
        
    Returns:
        Tuple of (nifty_csv_path, vix_csv_path)
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate filenames
    if consolidated:
        nifty_csv_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
        vix_csv_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nifty_csv_path = os.path.join(data_dir, f'nifty_data_{timestamp}.csv')
        vix_csv_path = os.path.join(data_dir, f'vix_data_{timestamp}.csv')
    
    # Save data
    nifty_df.to_csv(nifty_csv_path)
    vix_df.to_csv(vix_csv_path)
    
    print(f"Data saved to {nifty_csv_path} and {vix_csv_path}")
    return nifty_csv_path, vix_csv_path

def merge_dataframes(existing_df, new_df):
    """
    Merge existing and new dataframes, removing duplicates
    
    Args:
        existing_df: Existing DataFrame
        new_df: New DataFrame to merge
        
    Returns:
        Merged DataFrame
    """
    if existing_df is None or len(existing_df) == 0:
        return new_df
    
    if new_df is None or len(new_df) == 0:
        return existing_df
    
    # Ensure both dataframes have DatetimeIndex
    if not isinstance(existing_df.index, pd.DatetimeIndex):
        existing_df.index = pd.to_datetime(existing_df.index)
    
    if not isinstance(new_df.index, pd.DatetimeIndex):
        new_df.index = pd.to_datetime(new_df.index)
    
    # Make both timezone-naive for consistent comparison
    if existing_df.index.tz is not None:
        existing_df = existing_df.tz_localize(None)
    if new_df.index.tz is not None:
        new_df = new_df.tz_localize(None)
    
    # Concatenate and sort
    merged_df = pd.concat([existing_df, new_df])
    
    # Remove duplicates, keeping the latest data for each timestamp
    merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
    
    # Sort by datetime index
    merged_df = merged_df.sort_index()
    
    return merged_df

def consolidate_csv_files(data_dir='data'):
    """
    Consolidate all CSV files in the data directory into single files
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (nifty_df, vix_df) consolidated dataframes
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Find all CSV files
    nifty_files = glob.glob(os.path.join(data_dir, 'nifty_*.csv'))
    vix_files = glob.glob(os.path.join(data_dir, 'vix_*.csv'))
    
    # Initialize empty dataframes
    nifty_consolidated = pd.DataFrame()
    vix_consolidated = pd.DataFrame()
    
    # Process NIFTY files
    for file in nifty_files:
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            nifty_consolidated = merge_dataframes(nifty_consolidated, df)
            print(f"Added {len(df)} records from {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Process VIX files
    for file in vix_files:
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            vix_consolidated = merge_dataframes(vix_consolidated, df)
            print(f"Added {len(df)} records from {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Save consolidated files
    if not nifty_consolidated.empty and not vix_consolidated.empty:
        save_to_csv(nifty_consolidated, vix_consolidated, data_dir)
    
    return nifty_consolidated, vix_consolidated

def update_data(days=30, data_dir='data'):
    """
    Update existing data by fetching new data and merging
    
    Args:
        days: Number of days to fetch
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (nifty_df, vix_df) updated dataframes
    """
    # Try to load consolidated data first
    nifty_consolidated_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
    vix_consolidated_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
    
    if os.path.exists(nifty_consolidated_path) and os.path.exists(vix_consolidated_path):
        existing_nifty, existing_vix = load_from_csv(nifty_consolidated_path, vix_consolidated_path)
    else:
        # If no consolidated files, try to consolidate all files
        existing_nifty, existing_vix = consolidate_csv_files(data_dir)
    
    # Determine date range for fetching new data
    end_date = datetime.now(ZoneInfo("Asia/Kolkata"))
    
    if existing_nifty is not None and not existing_nifty.empty:
        # Get the latest date in the existing data
        latest_date = existing_nifty.index.max()
        
        # Set start_date to the day after the latest date in existing data
        start_date = latest_date + timedelta(days=1)
        
        # If the latest date is very recent, set start_date to days before end_date
        # to ensure some overlap for proper merging
        if (end_date - latest_date).days < 5:
            start_date = end_date - timedelta(days=days)
    else:
        # No existing data, fetch for the specified days
        start_date = end_date - timedelta(days=days)
    
    # Fetch new data
    new_nifty, new_vix = fetch_new_data(start_date, end_date)
    
    # Merge existing and new data
    updated_nifty = merge_dataframes(existing_nifty, new_nifty)
    updated_vix = merge_dataframes(existing_vix, new_vix)
    
    # Save updated data
    if updated_nifty is not None and updated_vix is not None:
        save_to_csv(updated_nifty, updated_vix, data_dir)
    
    return updated_nifty, updated_vix

def get_data_summary(data_dir='data'):
    """
    Get summary of available data
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with data summary
    """
    # Try to load consolidated data
    nifty_consolidated_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
    vix_consolidated_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
    
    summary = {
        'has_consolidated_files': False,
        'nifty_records': 0,
        'vix_records': 0,
        'nifty_date_range': None,
        'vix_date_range': None,
        'other_files': []
    }
    
    # Check for consolidated files
    if os.path.exists(nifty_consolidated_path) and os.path.exists(vix_consolidated_path):
        summary['has_consolidated_files'] = True
        
        # Load data to get summary
        nifty_df, vix_df = load_from_csv(nifty_consolidated_path, vix_consolidated_path)
        
        if nifty_df is not None:
            summary['nifty_records'] = len(nifty_df)
            summary['nifty_date_range'] = (nifty_df.index.min(), nifty_df.index.max())
        
        if vix_df is not None:
            summary['vix_records'] = len(vix_df)
            summary['vix_date_range'] = (vix_df.index.min(), vix_df.index.max())
    
    # List all other data files
    all_files = glob.glob(os.path.join(data_dir, '*.csv'))
    for file in all_files:
        if os.path.basename(file) not in ['nifty_data_consolidated.csv', 'vix_data_consolidated.csv']:
            summary['other_files'].append({
                'name': os.path.basename(file),
                'size': os.path.getsize(file),
                'modified': datetime.fromtimestamp(os.path.getmtime(file))
            })
    
    return summary

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
        # Try loading consolidated data first
        nifty_consolidated_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
        vix_consolidated_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
        
        if os.path.exists(nifty_consolidated_path) and os.path.exists(vix_consolidated_path):
            nifty_df, vix_df = load_from_csv(nifty_consolidated_path, vix_consolidated_path)
        else:
            # If no consolidated files, try to find latest files
            nifty_files = glob.glob(os.path.join(data_dir, 'nifty_*.csv'))
            vix_files = glob.glob(os.path.join(data_dir, 'vix_*.csv'))
            
            if nifty_files and vix_files:
                latest_nifty = max(nifty_files, key=os.path.getmtime)
                latest_vix = max(vix_files, key=os.path.getmtime)
                
                nifty_df, vix_df = load_from_csv(latest_nifty, latest_vix)
    
    # If loading from disk failed or was skipped, fetch new data
    if nifty_df is None or vix_df is None:
        nifty_df, vix_df = update_data(fetch_days, data_dir)
    
    return nifty_df, vix_df 