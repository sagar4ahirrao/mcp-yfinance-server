# 💹 MCP YFinance Stock Server

This project sets up a **stock Price server** powered by the **Yahoo Finance (YFinance)** API and built for seamless integration with **MCP (Multi-Agent Control Protocol)**.

It allows AI agents or clients to retrieve **real-time stock data**, manage a **watchlist**, and **analyze historical trends** with ease.

![image](https://github.com/user-attachments/assets/b7f27442-823e-45a1-8f57-e0f6282ee36f)

---
Before building this server you can try building this simple [crypto server](https://github.com/Adity-star/mcp-crypto-server)
---

## 📦 Step 1: Set Up the Environment with uv

We use **uv** — a modern, ultra-fast Python package manager — to manage our project environment.

### 🛠️ Installation & Setup

Run the following commands in your terminal:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh 

# Create and navigate to your project directory
mkdir mcp-yfinance-server
cd mcp-yfinance-server

# Initialize a new project
uv init

# Create and activate the virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
uv add "mcp[cli]" yfinance
```
## 🚀 Step 2: Running the MCP Server
After setting up the environment:
```bash
cp ../stock_price_serve.py .
uv run stock_price_server.py
```
📄 Check out [stock_price_server.py](https://github.com/Adity-star/mcp-yfinance-server/blob/main/stock_price_server.py) to explore how the server fetches and formats YFinance data.

# 🛠️ MCP Tool Reference

The MCP server exposes the following tools for financial research, portfolio tracking, and real-time market monitoring:

## 📦 Tool List

| Tool Name                      | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `add_to_watchlist`            | Add a stock ticker to your personal watchlist.                             |
| `remove_from_watchlist`       | Remove a stock from your watchlist.                                        |
| `get_watchlist`               | Retrieve the full list of watchlisted tickers.                             |
| `get_watchlist_prices`        | Fetch the most recent prices for all watchlisted tickers.                  |
| `get_realtime_watchlist_prices` | Get cached real-time prices (faster access).                              |
| `get_stock_price`             | Retrieve the current price for a given ticker symbol.                      |
| `get_stock_history`           | Download historical price data in CSV format for a ticker.                 |
| `compare_stocks`              | Compare two stock prices (useful for relative performance analysis).       |

## 🧠 Use Cases

These tools are ideal for:
- Portfolio tracking
- Financial research
- Real-time market reactions

---

> ⚙️ Keep this reference handy for building intelligent financial applications with the MCP server.

## 🔍 Step 3: Inspecting the MCP Server

You can test and inspect your MCP server using the **MCP Server Inspector**.  
Run the following command in your terminal:

```bash
$ mcp dev stock_price_server.py
```
![image](https://github.com/user-attachments/assets/886182a3-e996-4713-aec6-d9ab3fac3bd9)


## ⚙️ Step 4: Configure Your MCP Server

Update your MCP config with the following entry:

```json
{
  "mcpServers": {
    "yfinance-price-tracker": {
      "command": "/ABSOLUTE/PATH/TO/uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/YOUR/mcp-yfinance-server",
        "run",
        "stock_price_server.py"
      ]
    }
  }
}
```
🔁 Note: Replace /ABSOLUTE/PATH/TO/... with the actual paths on your local machine.
💡 Heads up: The tools may still appear under the server ID crypto-price-tracker. For better clarity and organization, consider renaming it to yfinance-price-tracker in your config.

## 🔁 Step 4: Restart Claude Desktop

Restart **Claude Desktop** (or any interface that uses MCP) to reload and activate your new YFinance tools.

---
## ✅ Step 5: Testing the MCP Server with Claude Desktop

With everything installed and configured, you're ready to test your MCP server in **Claude Desktop**.

### 🗣️ Try asking Claude:

> **"Could you please compare the stock prices of Tesla and Apple?"**  
> *Claude should respond with a relative performance comparison using your custom tools.*

> **"Can you provide the historical data for Tesla over the past month?"**  
> *Claude should fetch the CSV data using `get_stock_history`.*

> **"could you setup a watchlist for stocks apple, tesla and reliance?"**
> *Claude should setup a watch list for the given stock.*

### 📊 Visualize Stock Performance

You can also ask Claude to create visualizations, like:

> **"Create a chart showing Apple’s stock performance over the last 30 days."**


If configured correctly, Claude will retrieve historical data for Apple and generate a clean visualization of its recent trends.

Check the chart generated by my server [here](https://github.com/Adity-star/mcp-yfinance-server/blob/main/assets/Screenshot%20(16).png).

can also check out this [published link](https://claude.site/artifacts/bba1b878-de53-4988-a0b5-377f2d202b3a) of my chart.

---

🧪 These tests ensure your MCP integration is working end-to-end—from data retrieval to real-time analysis and visualization.



## 📫 Feedback & Contributions

Have an idea or found a bug?  
We welcome [issues](https://github.com/Adity-star/mcp-yfinance-server/issues) and [pull requests](https://github.com/Adity-star/mcp-yfinance-server/pulls) — your contributions make this project better!

---

## 📜 License

[MIT © 2025 Ak Aditya](https://github.com/Adity-star/mcp-yfinance-server/blob/main/LICENSE).



🚀 Let’s build better tools together.
