import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from prompt import PROMPT
from tools import get_historical_chart, get_stock_info, get_stock_news
import plotly.graph_objects as go

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

tools_list = [get_stock_info, get_historical_chart, get_stock_news]

model = genai.GenerativeModel(
    model_name='gemini-flash-latest', 
    tools=tools_list,
    system_instruction=PROMPT 
)

if "agent_chat" not in st.session_state:
    st.session_state.agent_chat = model.start_chat(enable_automatic_function_calling=True)

# Streamlit ui
st.set_page_config(page_title="Stock Insights AI Agent", layout="wide")
st.title("📈 Stock Insights AI Assistant")

query = st.text_input("Ask a specific question (e.g., 'What is the PE ratio of Apple?' or 'Show me Reliance news')")

if query:
    with st.spinner("AI Agent is reasoning and fetching data..."):
        response = st.session_state.agent_chat.send_message(query)
        
        st.info(response.text)

        for part in response.candidates[0].content.parts:
            if fn := part.function_call:
                ticker_arg = fn.args['ticker']
                st.divider()
                
                if fn.name == "get_stock_info":
                    st.write(f"### 📋 Fundamentals: {ticker_arg}")
                    data = get_stock_info(ticker_arg)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current Price", f"${data['Price']}")
                    c2.metric("PE Ratio", f"{data['PE Ratio']:.2f}" if data['PE Ratio'] else "N/A")
                    c3.metric("Market Cap", f"{data['Market Cap']:,}" if data['Market Cap'] else "N/A")

                elif fn.name == "get_historical_chart":
                    st.write(f"### 📊 Price Trend: {ticker_arg}")
                    chart_data = get_historical_chart(ticker_arg)
                    st.plotly_chart(fig, use_container_width=True)

                elif fn.name == "get_stock_news":
                    st.write(f"### 📰 Latest News: {ticker_arg}")
                    news_data = get_stock_news(ticker_arg)
                    for n in news_data:
                        with st.expander(n['title']):
                            st.write(f"Source: {n['source']}")
                            st.write(f"[Read Article]({n['link']})")