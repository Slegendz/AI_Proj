import yfinance as yf
import plotly.graph_objects as go

def get_stock_info(ticker: str):
    """Fetches Price, PE Ratio, Market Cap, and EPS."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Price": info.get("currentPrice"),
        "Prev Close": info.get("previousClose"),
        "PE Ratio": info.get("trailingPE"),
        "Market Cap": info.get("marketCap"),
        "EPS": info.get("trailingEps")
    }

def get_historical_chart(ticker: str):
    """Generates a 1-month Price Trend chart."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    fig = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name=ticker)])
    fig.update_layout(title=f"{ticker} Trend", xaxis_title="Date", yaxis_title="Price")
    return fig

def get_stock_news(ticker: str):
    """Fetches the latest 3 stock news items."""
    stock = yf.Ticker(ticker)
    news = []
    for item in stock.news[:3]:
        content = item.get('content', {})
        news.append({
            "title": content.get('summary', item.get('title')),
            "source": content.get('provider', {}).get('displayName', 'Unknown'),
            "link": content.get('canonicalUrl', {}).get('url', item.get('link'))
        })
    return news