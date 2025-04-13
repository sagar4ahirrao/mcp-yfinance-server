# 💹 MCP YFinance Stock Server
![Python](https://img.shields.io/badge/python-3.10-blue)
![MCP](https://img.shields.io/badge/MCP-Compatible-green)
![License](https://img.shields.io/github/license/Adity-star/mcp-yfinance-server)


This project sets up a **stock Price server** powered by the **Yahoo Finance (YFinance)** API and built for seamless integration with **MCP (Multi-Agent Control Protocol)**.

It allows AI agents or clients to retrieve **real-time stock data**, manage a **watchlist**, and **analyze historical trends** with ease.

![image](https://github.com/user-attachments/assets/b7f27442-823e-45a1-8f57-e0f6282ee36f)

---
## 🪙 Start Simple: Build a Crypto Price Tracker First
Before diving into the full-blown stock server, I recommend starting with this simple crypto tracker built with Python + MCP 👇

🔗 GitHub Repo:
https://github.com/Adity-star/mcp-crypto-server

> You'll learn how to:
> - Use MCP to expose crypto tools like get_price("BTC")
> - Build an API with FastAPI
> - Fetch real-time prices using the Alpaca API

---
## 📈 Then Level Up: Build the yFinance Stock Server
Once you're familiar with the flow, move on to this more advanced stock tracker 💹

🔗 GitHub Repo:
[https://github.com/Adity-star/mcp-yfinance-server](https://github.com/Adity-star/mcp-yfinance-server)

📝 Detailed Blog:
👉 [How I Built My Own Stock Server with Python, yFinance, and a Touch of Nerdy Ambition](https://medium.com/@aakuskar.980/how-i-built-my-own-stock-server-with-python-yfinance-and-a-touch-of-nerdy-ambition-b562dc1d7b93)

> Includes:
> - Watchlists
> - Historical comparisons
> - Real-time(ish) updates
> - A full-featured dashboard

---

## 📦 Step 1: Set Up the Environment (with uv)

We use **[uv](https://github.com/astral-sh/uv)** — a modern, ultra-fast Python package manager — to manage our project environment.

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
---
## 🚀 Step 2: Running the MCP Server
After setting up the environment:
```bash
cp ../stock_price_serve.py .
uv run stock_price_server.py
```
📄 Want to explore how it works? Check the full code here: 

📂 [stock_price_server.py](https://github.com/Adity-star/mcp-yfinance-server/blob/main/stock_price_server.py)

---

# 🛠️ MCP Tool Reference

The server exposes these tools for AI agents and CLI users:

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


> ⚙️ Keep this reference handy for building intelligent financial applications with the MCP server.

---

## 🔍 Step 3: Inspecting the MCP Server

You can test and inspect your MCP server using the **MCP Server Inspector**.  
Run the following command in your terminal:

```bash
$ mcp dev stock_price_server.py
```
![image](https://github.com/user-attachments/assets/886182a3-e996-4713-aec6-d9ab3fac3bd9)

---

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
⚠️ Replace /ABSOLUTE/PATH/TO/... with actual file paths.
💡 Tip: Rename your server from crypto-price-tracker to yfinance-price-tracker for clarity.

---
## 🔁 Step 5: Restart Claude Desktop

Restart **Claude Desktop** (or any interface that uses MCP) to reload and activate your new YFinance tools.

---
## ✅ Step 6: Testing the MCP Server with Claude Desktop

- With everything installed and configured, you're ready to test your MCP server in **Claude Desktop**.

Try these prompts in Claude Desktop:

> "Compare the stock prices of Tesla and Apple."
> → Uses compare_stocks

> "Get the historical data for Tesla over the past month."
> → Uses get_stock_history

> "Add Apple, Tesla, and Reliance to my watchlist."
> → Uses add_to_watchlist

> "Show me a chart of Apple’s stock over the last 30 days."
> → Claude can fetch + visualize using your server.

🖼 [View Sample Chart](https://github.com/Adity-star/mcp-yfinance-server/blob/main/assets/Screenshot%20(16).png)

🌐 [Published Link (Claude.site)](https://claude.site/artifacts/bba1b878-de53-4988-a0b5-377f2d202b3a)

🧪 These tests ensure your MCP integration is working end-to-end—from data retrieval to real-time analysis and visualization.

---

## 📫 Feedback & Contributions

- Have an idea or found a bug?  
- We welcome [issues](https://github.com/Adity-star/mcp-yfinance-server/issues) and [pull requests](https://github.com/Adity-star/mcp-yfinance-server/pulls)
- your contributions make this project better!

---
## 💬 Spread the Word
If this saved you from API rate limits or overpriced SaaS tools…

🌟 Star the repo
🍴 Fork it and build your own crypto/stock tool
📲 Tag me on X @AdityaAkuskar — I’d love to see what you build!

---

## 📜 License

[MIT © 2025 Ak Aditya](https://github.com/Adity-star/mcp-yfinance-server/blob/main/LICENSE).

---
🚀 Let’s build better tools together.

If you’d like a tweet thread, carousel, or launch post for this — I’ve got your back 😎
