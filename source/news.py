import yfinance as yf
from textblob import TextBlob
from typing import List, Dict
from datetime import datetime


def fetch_yahoo_news(symbol: str) -> List[Dict]:
    """
    Fetch and format recent news for a stock symbol from Yahoo Finance.
    Includes thumbnail image if available.
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        news_items = ticker.news[:5]  # Limit to 5 most recent
        formatted_news = []

        for item in news_items:
            thumbnail = ""
            if "thumbnail" in item and "resolutions" in item["thumbnail"]:
                resolutions = item["thumbnail"]["resolutions"]
                if resolutions:
                    thumbnail = resolutions[-1].get("url", "")

            formatted_news.append({
                "title": item.get("title", "No title"),
                "link": item.get("link", ""),
                "publisher": item.get("publisher", "Unknown"),
                "time": datetime.fromtimestamp(item["providerPublishTime"]).strftime('%Y-%m-%d %H:%M') \
                        if "providerPublishTime" in item else "N/A",
                "thumbnail": thumbnail
            })

        return formatted_news
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return []


def analyze_sentiment(headlines: List[str]) -> Dict[str, float]:
    """
    Analyze sentiment of headlines using TextBlob.
    """
    if not headlines:
        return {"polarity": 0.0, "subjectivity": 0.0}

    total_polarity = 0
    total_subjectivity = 0
    for headline in headlines:
        blob = TextBlob(headline)
        total_polarity += blob.sentiment.polarity
        total_subjectivity += blob.sentiment.subjectivity

    count = len(headlines)
    return {
        "polarity": round(total_polarity / count, 3),
        "subjectivity": round(total_subjectivity / count, 3)
    }


def get_company_news(symbol: str) -> Dict:
    """
    Get structured news headlines for a company.
    """
    return {
        "symbol": symbol.upper(),
        "news": fetch_yahoo_news(symbol)
    }


def get_news_and_sentiment(symbol: str) -> Dict:
    """
    Get structured news headlines and sentiment analysis.
    """
    raw_news = fetch_yahoo_news(symbol)
    headlines = [item["title"] for item in raw_news]
    sentiment = analyze_sentiment(headlines)
    return {
        "symbol": symbol.upper(),
        "news": raw_news,
        "sentiment": sentiment
    }
