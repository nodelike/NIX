# AlphaZero India VIX Trader

An AlphaZero-inspired trading system for the NIFTY index based on VIX signals. This system uses deep reinforcement learning with Monte Carlo Tree Search (MCTS) to make daily trading decisions at 9:05 AM.

## Features

- **AlphaZero Architecture**: Deep neural networks with MCTS for trading decisions
- **Interactive UI**: Streamlit-based dashboard for visualization and model training
- **Backtesting**: Comprehensive backtesting capabilities with performance metrics
- **9:05 AM Trading**: Focused on trading at market open for optimal execution

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/indiavix.git
   cd indiavix
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

The system can be used in several modes:

### Web Application

Run the web application for an interactive experience:

```
python main.py --mode app
```

This will launch a Streamlit application with dashboard, training, backtesting, and prediction pages.

### Fetch Data

To fetch the latest market data:

```
python main.py --mode fetch_data --days 30
```

This will download 30 days of 1-minute data for NIFTY and India VIX.

### Train Model

To train the AlphaZero model:

```
python main.py --mode train --episodes 10 --batches 20
```

This will run 10 self-play episodes and 20 training batches.

### Run Backtest

To backtest the trading strategy:

```
python main.py --mode backtest
```

## System Architecture

The project is organized into the following components:

- **src/alphazero**: Core AlphaZero implementation
  - `model.py`: Neural network architecture
  - `mcts.py`: Monte Carlo Tree Search
  - `environment.py`: Trading environment
  - `trader.py`: Trading system

- **src/data_processing**: Data handling
  - `data_loader.py`: Data loading utilities
  - `features.py`: Feature extraction

- **app**: Web application
  - `app.py`: Streamlit dashboard

## Trading Strategy

The system learns to make trading decisions (buy, sell, or hold) for the NIFTY index at 9:05 AM every day. It uses the following approach:

1. Analyze market patterns from previous days
2. Extract features from NIFTY and VIX data
3. Use neural networks to predict optimal actions
4. Enhance decisions with Monte Carlo Tree Search
5. Update strategy through reinforcement learning

## Performance Metrics

The system evaluates performance using:

- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AlphaZero by DeepMind for the reinforcement learning architecture
- Streamlit for the web application framework
- Yahoo Finance for market data