# AlphaZero Trader for India VIX

An AlphaZero-based trading system for the Indian markets, focusing on NIFTY and India VIX.

## Features

- **AlphaZero Algorithm**: Trading strategy based on the AlphaZero reinforcement learning algorithm
- **Technical Analysis**: Feature extraction from NIFTY and India VIX data
- **Interactive UI**: Flask-based dashboard for visualization and model training
- **Backtesting**: Built-in backtesting system to evaluate trading strategies

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/indiavix.git
   cd indiavix
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Run the setup script to install dependencies and create necessary directories:
   ```
   python setup_environment.py
   ```

   Or manually install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure environment variables by editing the `.env` file (created from `.env.example`).

## Usage

Run the application with:

```
python app.py
```

This will launch a Flask application with dashboard, training, backtesting, and prediction pages. By default, the application will be available at http://127.0.0.1:8080.

### Command Line Interface

You can also use the command line interface:

```
python main.py --mode [dashboard|train|backtest] [--update] [--days 30]
```

Options:
- `--mode`: Mode to run (dashboard, train, or backtest)
- `--update`: Update market data before running
- `--days`: Number of days to fetch data for (when updating)

Examples:
```
# Run the dashboard
python main.py --mode dashboard

# Update data (30 days) and run backtesting
python main.py --mode backtest --update --days 30

# Train the model with existing data
python main.py --mode train
```

## Configuration and Settings

The application uses a multi-level configuration system:

### Environment Variables

Basic configuration is loaded from the `.env` file, which is created from `.env.example` during setup. You can modify these values to adjust application behavior:

- `FLASK_ENV`: Set to `development` or `production`
- `PORT`: The port number for the Flask server (default: 8080)
- `SECRET_KEY`: Secret key for session encryption (auto-generated if not provided)
- `DEBUG`: Enable Flask debug mode when set to `true`

### User Settings

User settings can be configured through the Settings page in the web interface. These settings affect trading and are stored both in your session and in a persistent configuration file:

- **Lot Size**: Number of shares/contracts per trade
- **Initial Capital**: Starting capital for backtesting and simulation
- **Trading Time**: The time of day to execute trades (format: HH:MM)

The settings are saved to `instance/user_config.ini` and automatically loaded each time the application starts. If you modify settings in the web interface, they will be immediately available to all components of the application.

To reset settings to their default values, use the "Reset Settings" button on the Settings page.

## Project Structure

```
indiavix/
├── app/                    # Flask application
│   ├── __init__.py         # App initialization
│   ├── routes/             # Flask route controllers
│   ├── static/             # Static assets (CSS, JS)
│   └── templates/          # Jinja2 templates
├── data/                   # Data directory
│   ├── nifty_data_consolidated.csv  # NIFTY historical data
│   └── vix_data_consolidated.csv    # India VIX historical data
├── models/                 # Saved models
├── src/                    # Source code
│   ├── alphazero/          # AlphaZero implementation
│   ├── data_processing/    # Data processing utilities
│   └── utils/              # Utility functions
├── instance/               # Flask instance folder (sessions, etc.)
│   ├── flask_session/      # Flask session files
│   └── user_config.ini     # User configuration file
├── config.py               # Configuration management
├── main.py                 # CLI entry point
├── app.py                  # Flask app entry point
├── .env                    # Environment variables (created from .env.example)
├── .env.example            # Example environment variables
└── requirements.txt        # Dependencies
```

## Data Sources

- **NIFTY 50 Index**: Historical data from Yahoo Finance (^NSEI)
- **India VIX**: Historical data from Yahoo Finance (^INDIAVIX)

## Development

### Dependencies

- Flask for the web application framework
- TensorFlow for the neural network
- Pandas for data manipulation
- Plotly for interactive charts

## License

This project is licensed under the MIT License - see the LICENSE file for details.