import streamlit as st
import plotly.express as px
import os
import openai
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np # linear algebra
import pandas as pd # for ticker preparation
import nltk
import PIL
import plotly.express as px # for ticker visualization
from textblob import TextBlob # for sentiment analysis
from plotly.subplots import make_subplots
import yfinance as yf
import datetime
import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
import glob
st.set_option('deprecation.showPyplotGlobalUse', False)
import seaborn as sns
#%matplotlib inline

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf
from datetime import date, timedelta
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from scipy import stats
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import plotly.offline as pyo
sns.set_style('darkgrid')
pyo.init_notebook_mode()
nltk.download('vader_lexicon')
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import random
from itertools import count
from nltk.util import pr
import re
import string
import logging



ima= PIL.Image.open('C:/Users/PC/Downloads/rosaline/assets/thumb.webp')
bon= PIL.Image.open('C:/Users/PC/Downloads/rosaline/assets/big.jpg')






import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

# Main Streamlit app
def main():
    st.title("Bespoke Stocks Analysis")

    # User input
    user = st.text_input("Enter a prompt Analysis")
    ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL)")
    button = st.button("**Generate:**")

    if ticker and button:
        # Fetch stock data from Yahoo Finance
        stock_data = yf.download(ticker)

        today = date.today()
        d1 = today.strftime("%Y-%m-%d")
        end_date = d1
        d2 = date.today() - timedelta(days=365)
        d2 = d2.strftime("%Y-%m-%d")
        start_date = d2

        # Get historical stock prices
        data = stock_data.loc[start_date:end_date]

        # Display candlestick chart
        figure1 = go.Figure(data=[go.Candlestick(x=data.index,
                                                 open=data['Open'],
                                                 high=data['High'],
                                                 low=data['Low'],
                                                 close=data['Close'])])
        figure1.update_layout(title="Stock Price Analysis", xaxis_rangeslider_visible=False)
        st.plotly_chart(figure1)

        # Display bar chart
        figure2 = px.bar(data, x=data.index, y=data['Close'], labels={'x': 'Date', 'y': 'Close'})
        st.plotly_chart(figure2)

        # Display line chart with rangeslider
        figure3 = px.line(data, x=data.index, y=data['Close'],
                          title='Stock Market Analysis with Rangeslider')
        figure3.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(figure3)

        # Display line chart with time period selectors
        figure4 = px.line(data, x=data.index, y=data['Close'],
                          title='Stock Market Analysis with Time Period Selectors')
        figure4.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        st.plotly_chart(figure4)

        # Range slider for selecting data range
        st.subheader("Select Data Range")
        start_date = st.date_input("Start Date", value=data.index.min())
        end_date = st.date_input("End Date", value=data.index.max())

        # Filter stock prices based on selected data range
        filtered_data = data.loc[start_date:end_date]

        # Display summary statistics
        st.subheader("Summary Statistics")
        st.write(filtered_data.describe())

        # Create figure with range selector
        fig = go.Figure()




                        
       # Add line chart of stock prices
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'], name='Stock Prices'))

        # Update layout with range selector
        fig.update_layout(
            title="Historical Stock Prices",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

        # Display line chart with range selector
        st.plotly_chart(fig)

# Run the app
if __name__ == '__main__':
    main()
