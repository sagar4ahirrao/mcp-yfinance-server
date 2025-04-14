import yfinance as yf
from technical_indicators import TechnicalIndicators
from mcp.server.fastmcp import FastMCP
import threading
import time
import asyncio
from news import get_company_news, get_news_and_sentiment

# Create the MCP server instance
mcp = FastMCP("Stock Price Server")

# In-memory watchlist and real-time price cache
watchlist = set()
watchlist_prices = {}

# --- Utility Functions ---
def fetch_ticker(symbol: str):
    """Helper to safely fetch a yfinance Ticker."""
    return yf.Ticker(symbol.upper())

def safe_get_price(ticker) -> float:
    """Attempt to retrieve the current price of a stock."""
    try:
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        price = ticker.info.get('regularMarketPrice')
        if price is not None:
            return float(price)
        raise ValueError("Price data not available.")
    except Exception as e:
        raise ValueError(f"Error retrieving stock price: {e}")


# Import the TechnicalIndicators class here
ti = TechnicalIndicators()

@mcp.tool()
def get_stock_price(symbol: str) -> float:
    """
    Retrieve the current stock price for the given ticker symbol.
    Returns the latest closing price as a float.
    """
    symbol = symbol.upper()
    ticker = fetch_ticker(symbol)
    return safe_get_price(ticker)

@mcp.tool()
def get_technical_summary(symbol: str) -> dict:
    """
    Generate a complete technical analysis summary for a stock using the TechnicalIndicators class.
    """
    return ti.get_technical_summary(symbol)

@mcp.resource("stock://{symbol}")
def stock_resource(symbol: str) -> str:
    """
    Expose stock price data as a resource.
    Returns a formatted string with the current stock price.
    """
    try:
        price = get_stock_price(symbol)
        return f"The current price of {symbol.upper()} is ${price:.2f}"
    except ValueError as e:
        return f"[{symbol.upper()}] Error: {e}"

@mcp.tool()
def get_stock_history(symbol: str, period: str = "1mo") -> str:
    """
    Retrieve historical stock data in CSV format.
    """
    symbol = symbol.upper()
    try:
        ticker = fetch_ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return f"[{symbol}] No historical data found for period '{period}'."
        return data.to_csv()
    except Exception as e:
        return f"[{symbol}] Error fetching historical data: {e}"

@mcp.tool()
def compare_stocks(symbol1: str, symbol2: str) -> str:
    """
    Compare two stock prices.
    """
    symbol1, symbol2 = symbol1.upper(), symbol2.upper()
    try:
        price1 = get_stock_price(symbol1)
        price2 = get_stock_price(symbol2)
        if price1 > price2:
            return f"{symbol1} (${price1:.2f}) is higher than {symbol2} (${price2:.2f})."
        elif price1 < price2:
            return f"{symbol1} (${price1:.2f}) is lower than {symbol2} (${price2:.2f})."
        else:
            return f"{symbol1} and {symbol2} have the same price (${price1:.2f})."
    except Exception as e:
        return f"Error comparing stocks: {e}"

# --- Watchlist Management ---
@mcp.tool()
def add_to_watchlist(symbol: str) -> str:
    symbol = symbol.upper()
    watchlist.add(symbol)
    return f"[Watchlist] Added {symbol}."

@mcp.tool()
def remove_from_watchlist(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol in watchlist:
        watchlist.remove(symbol)
        return f"[Watchlist] Removed {symbol}."
    return f"[Watchlist] {symbol} was not in the list."

@mcp.tool()
def get_watchlist() -> list:
    return sorted(watchlist)

@mcp.tool()
def get_watchlist_prices() -> dict:
    """
    Get the most recent prices for all stocks in the watchlist.
    """
    prices = {}
    for symbol in sorted(watchlist):
        try:
            prices[symbol] = round(get_stock_price(symbol), 2)
        except Exception as e:
            prices[symbol] = f"Error: {e}"
    return prices

# --- Simulated Real-Time Updates ---
def update_prices():
    """
    Background thread to update watchlist prices every 30 seconds.
    """
    while True:
        for symbol in list(watchlist):  # Use list to avoid set size change errors
            try:
                ticker = fetch_ticker(symbol)
                watchlist_prices[symbol] = round(safe_get_price(ticker), 2)
            except Exception as e:
                watchlist_prices[symbol] = f"Error: {e}"
        time.sleep(30)

@mcp.tool()
def get_realtime_watchlist_prices() -> dict:
    """
    Get real-time cached prices from the background updater.
    """
    return dict(sorted(watchlist_prices.items()))

@mcp.tool()
def get_company_news_headlines(symbol: str) -> dict:
    """
    Get latest news headlines for a company using Yahoo Finance.
    """
    return get_company_news(symbol)

@mcp.tool()
def get_stock_news_sentiment(symbol: str) -> dict:
    """
    Get recent news headlines and sentiment analysis for a stock.
    """
    return get_news_and_sentiment(symbol)


# --- Start the background price update thread ---
price_update_thread = threading.Thread(target=update_prices, daemon=True)
price_update_thread.start()

# Run the server
if __name__ == "__main__":
    mcp.run()
