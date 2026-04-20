import os
import streamlit as st
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Stock price and fundamentals
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "Price": info.get("currentPrice"),
        "Prev Close": info.get("previousClose"),
        "PE Ratio": info.get("trailingPE"),
        "Market Cap": info.get("marketCap"),
        "EPS": info.get("trailingEps")
    }

# Historical data and visualizations
def get_historical_chart(ticker, period="1mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    fig = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name=ticker)])
    fig.update_layout(title=f"{ticker} Stock Trend", xaxis_title="Date", yaxis_title="Price (USD)")
    return fig

# Stock news
def get_stock_news(ticker):
    stock = yf.Ticker(ticker)
    return stock.news[:3] 

# Ai agent
def financial_agent(user_query):
    model = genai.GenerativeModel('gemini-flash-latest')
    
    # prompt = f"""
    # You are a financial AI assistant. Analyze the user query: "{user_query}"
    # Identify the ticker symbol (e.g., AAPL, RELIANCE.NS) and the intent.
    # Intent must be one of : 'price', 'chart', 'news', 'fundamentals'.
    # Respond ONLY in JSON format: {{"ticker": "SYMBOL", "intent": "INTENT"}}
    # """

    prompt = f"""
    You are a financial AI assistant. Analyze the user query: "{user_query}"
    Identify the ticker symbol (e.g., AAPL, RELIANCE.NS).
    Respond ONLY in JSON format: {{"ticker": "SYMBOL" }}
    """
    
    response = model.generate_content(prompt)
    return eval(response.text)

# Streamlit
st.set_page_config(page_title="Stock Insights AI Agent")
st.title("📈 Stock Insights AI Assistant")  

query = st.text_input("Ask about a stock (e.g., 'Show chart of Reliance' or 'Price of Apple')") 

if query:
    with st.spinner("AI is thinking..."):
        decision = financial_agent(query)
        ticker = decision['ticker']
        # intent = decision['intent']
        
        st.subheader(f"Results for {ticker}")
        
        # if intent == 'price' or intent == 'fundamentals':
        #     data = get_stock_info(ticker)
        #     col1, col2, col3 = st.columns(3)
        #     col1.metric("Current Price", f"${data['Price']}")
        #     col2.metric("PE Ratio", data['PE Ratio'])
        #     col3.metric("Market Cap", f"{data['Market Cap']:,}")
            
        # elif intent == 'chart':
        #     fig = get_historical_chart(ticker)
        #     st.plotly_chart(fig)
            
        # elif intent == 'news':
        #     news_items = get_stock_news(ticker)
        #     for item in news_items:
        #         st.write(f"**{item['title']}**")
        #         st.write(f"Source: {item['publisher']} | [Link]({item['link']})")

        data = get_stock_info(ticker)
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${data['Price']}")
        col2.metric("PE Ratio", data['PE Ratio'])
        col3.metric("Market Cap", f"{data['Market Cap']:,}")
        
        fig = get_historical_chart(ticker)
        st.plotly_chart(fig)
        
        news_items = get_stock_news(ticker)
        for item in news_items:
            content = item['content']
            st.write(f"**{content['summary']}**")
            st.write(f"Source: {content['provider']['displayName']} | [Link]({content['canonicalUrl']['url']})")