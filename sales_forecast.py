import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import streamlit as st

def load_data():
    dates = pd.date_range(start="2022-01-01", periods=200)
    sales = np.random.randint(100, 500, size=(200,))
    df = pd.DataFrame({"Date": dates, "Sales": sales})
    df['Day'] = np.arange(len(df))
    return df

def train_model(df):
    X = df[['Day']]
    y = df['Sales']
    model = RandomForestRegressor().fit(X, y)
    return model

def dashboard():
    st.title("📊 AI Sales Forecast Dashboard")
    df = load_data()
    model = train_model(df)
    future = pd.DataFrame({'Day': np.arange(len(df), len(df)+30)})
    preds = model.predict(future)
    st.line_chart(pd.concat([df[['Sales']], pd.DataFrame({"Sales": preds}, index=range(len(df), len(df)+30))]))
    st.write("🔮 Next 30-day Sales Forecast Shown Above")

if __name__ == '__main__':
    dashboard()