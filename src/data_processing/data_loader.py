import pandas as pd
import yfinance as yf
import os
import glob
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def load_from_csv(nifty_csv_path, vix_csv_path):
    """
    Load NIFTY and VIX data from CSV files with consistent alignment
    
    Args:
        nifty_csv_path: Path to NIFTY CSV file
        vix_csv_path: Path to VIX CSV file
        
    Returns:
        Tuple of (nifty_df, vix_df)
    """
    try:
        if not os.path.exists(nifty_csv_path):
            print(f"NIFTY data file not found: {nifty_csv_path}")
            return None, None
            
        if not os.path.exists(vix_csv_path):
            print(f"VIX data file not found: {vix_csv_path}")
            return None, None
            
        # Load data with proper error handling
        try:
            nifty_df = pd.read_csv(nifty_csv_path, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Error reading NIFTY data file: {e}")
            return None, None
            
        try:
            vix_df = pd.read_csv(vix_csv_path, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Error reading VIX data file: {e}")
            return None, None
        
        # Process the data using our centralized function
        return process_market_data(nifty_df, vix_df)
    
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        import traceback
        traceback.print_exc()
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
        
        # Ensure both dates have the same timezone
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        
        print(f"Fetching data from {start_date.date()} to {end_date.date()}")
        
        # Get NIFTY data
        nifty_df = get_chunked_data("^NSEI", start_date, end_date)
        
        # Get India VIX data
        vix_df = get_chunked_data("^INDIAVIX", start_date, end_date)
        
        if nifty_df is None or vix_df is None:
            print("Error: Failed to fetch market data.")
            return None, None
            
        if nifty_df.empty or vix_df.empty:
            print("Warning: Fetched data is empty.")
            return nifty_df, vix_df
            
        print(f"Fetched {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records")
        
        # Process and align the data
        return process_market_data(nifty_df, vix_df)
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()
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
        print(f"Warning: No data retrieved for {ticker_symbol}")
        # Return an empty DataFrame with expected columns instead of raising an error
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
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
    try:
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Try to load consolidated data first
        nifty_consolidated_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
        vix_consolidated_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
        
        # Check if we already have data files
        have_existing_data = os.path.exists(nifty_consolidated_path) and os.path.exists(vix_consolidated_path)
        
        # Set up dates for fetching
        end_date = datetime.now(ZoneInfo("Asia/Kolkata"))
        
        if have_existing_data:
            # Load existing data
            existing_nifty, existing_vix = load_from_csv(nifty_consolidated_path, vix_consolidated_path)
            
            if existing_nifty is None or existing_vix is None:
                print("Error loading existing data files. Will fetch new data.")
                start_date = end_date - timedelta(days=days)
            else:
                # Get latest date from existing data
                latest_date = max(existing_nifty.index.max(), existing_vix.index.max())
                
                # Fetch from the day after our latest date
                start_date = latest_date + timedelta(days=1)
                
                # If latest date is today, no need to update
                if latest_date.date() >= end_date.date():
                    print("Data already up to date. No new data to fetch.")
                    return existing_nifty, existing_vix
        else:
            # No existing data, fetch for the requested number of days
            start_date = end_date - timedelta(days=days)
            existing_nifty, existing_vix = None, None
        
        # Ensure both dates have the same timezone for comparison
        if start_date.tzinfo is None and end_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=end_date.tzinfo)
        elif start_date.tzinfo is not None and end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=start_date.tzinfo)
            
        # Fetch new data if needed
        if (end_date - start_date).days < 1:
            print("No new data to fetch.")
            return existing_nifty, existing_vix
            
        print(f"Fetching new data from {start_date.date()} to {end_date.date()}")
        new_nifty, new_vix = fetch_new_data(start_date, end_date)
        
        # Check if we actually got new data
        if new_nifty is None or new_vix is None:
            print("No new data fetched. Fetch operation returned None.")
            return existing_nifty, existing_vix
            
        if len(new_nifty) == 0 or len(new_vix) == 0:
            print("New data is empty.")
            return existing_nifty, existing_vix
            
        # If we have existing data, merge with new data
        if existing_nifty is not None and existing_vix is not None:
            print("Merging new data with existing data...")
            merged_nifty = merge_dataframes(existing_nifty, new_nifty)
            merged_vix = merge_dataframes(existing_vix, new_vix)
        else:
            merged_nifty = new_nifty
            merged_vix = new_vix
        
        # Process and align the merged data
        processed_nifty, processed_vix = process_market_data(merged_nifty, merged_vix)
        
        # Save updated data
        if processed_nifty is not None and processed_vix is not None:
            save_to_csv(processed_nifty, processed_vix, data_dir, consolidated=True)
            
        return processed_nifty, processed_vix
        
    except Exception as e:
        print(f"Error updating data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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

def process_market_data(nifty_df, vix_df):
    """
    Process and align market data from any source
    
    Args:
        nifty_df: DataFrame with NIFTY data
        vix_df: DataFrame with VIX data
        
    Returns:
        Tuple of processed (nifty_df, vix_df)
    """
    try:
        if nifty_df is None or vix_df is None:
            print("Error: One or both dataframes are None")
            return None, None
            
        if len(nifty_df) == 0 or len(vix_df) == 0:
            print("Error: One or both dataframes are empty")
            return None, None
        
        # Ensure DatetimeIndex
        if not isinstance(nifty_df.index, pd.DatetimeIndex):
            nifty_df.index = pd.to_datetime(nifty_df.index)
            
        if not isinstance(vix_df.index, pd.DatetimeIndex):
            vix_df.index = pd.to_datetime(vix_df.index)
        
        # Normalize timezone handling
        # First, make sure both have the same timezone (or no timezone)
        if nifty_df.index.tz is None and vix_df.index.tz is not None:
            nifty_df.index = nifty_df.index.tz_localize(vix_df.index.tz)
        elif nifty_df.index.tz is not None and vix_df.index.tz is None:
            vix_df.index = vix_df.index.tz_localize(nifty_df.index.tz)
        elif nifty_df.index.tz is None and vix_df.index.tz is None:
            # If both are timezone-naive, localize to Asia/Kolkata
            nifty_df.index = nifty_df.index.tz_localize('Asia/Kolkata')
            vix_df.index = vix_df.index.tz_localize('Asia/Kolkata')
            
        # Make sure both dataframes are sorted by index
        nifty_df = nifty_df.sort_index()
        vix_df = vix_df.sort_index()
        
        # Find common dates between both dataframes
        common_dates = nifty_df.index.intersection(vix_df.index)
        
        # If there are no common dates, return None
        if len(common_dates) == 0:
            print("Error: No common dates between NIFTY and VIX data.")
            return None, None
            
        # Use only the common dates for both dataframes
        nifty_df = nifty_df.loc[common_dates]
        vix_df = vix_df.loc[common_dates]
        
        # Perform an additional check to ensure both dataframes have the same length
        if len(nifty_df) != len(vix_df):
            print(f"Warning: Length mismatch after alignment. NIFTY: {len(nifty_df)}, VIX: {len(vix_df)}")
            
            # Use the minimum length to ensure they match
            min_length = min(len(nifty_df), len(vix_df))
            nifty_df = nifty_df.iloc[:min_length]
            vix_df = vix_df.iloc[:min_length]
        
        # Ensure consistent columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in nifty_df.columns:
                if col == 'Volume':
                    nifty_df[col] = 0  # Add default volume if missing
                else:
                    print(f"Error: Required column '{col}' missing from NIFTY data")
                    return None, None
                    
            if col not in vix_df.columns:
                if col == 'Volume':
                    vix_df[col] = 0  # Add default volume if missing
                else:
                    print(f"Error: Required column '{col}' missing from VIX data")
                    return None, None
        
        # Finally, make both timezone-naive for consistency with the rest of the code
        if nifty_df.index.tz is not None:
            nifty_df = nifty_df.tz_localize(None)
        if vix_df.index.tz is not None:
            vix_df = vix_df.tz_localize(None)
        
        print(f"Successfully processed {len(nifty_df)} records")
        return nifty_df, vix_df
        
    except Exception as e:
        print(f"Error processing market data: {e}")
        import traceback
        traceback.print_exc()
        return None, None 