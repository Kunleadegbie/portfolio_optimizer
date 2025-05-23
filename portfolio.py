import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fpdf import FPDF

# Page config
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# Sidebar Inputs
st.sidebar.title("📊 Portfolio Optimizer")

tickers_input = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL,MSFT,TSLA,NVDA,AMZN")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
rf_rate = st.sidebar.number_input("Risk-free rate (%)", value=0.0)
opt_metric = st.sidebar.selectbox("Optimize for:", ['Sharpe Ratio', 'Sortino Ratio'])
optimize_button = st.sidebar.button("🚀 Optimize Portfolio")

# Fetch Data
def get_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)

    st.write("✅ Downloaded data preview:")
    st.write(data.head())
    st.write("Columns:", data.columns)

    if data.empty:
        raise ValueError("No data found for the provided tickers and date range.")

    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            prices = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        else:
            raise ValueError("'Adj Close' or 'Close' not found in multi-ticker data.")
    else:
        # single ticker case
        if 'Adj Close' in data.columns:
            prices = data[['Adj Close']]
            prices.columns = [tickers if isinstance(tickers, str) else tickers[0]]
        elif 'Close' in data.columns:
            prices = data[['Close']]
            prices.columns = [tickers if isinstance(tickers, str) else tickers[0]]
        else:
            raise ValueError("'Adj Close' or 'Close' not found in single-ticker data.")

    return prices

# Portfolio Performance Metrics
def portfolio_performance(weights, mean_returns, cov_matrix, rf_rate, daily_returns, metric):
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    portfolio_daily_returns = daily_returns.dot(weights)

    if metric == 'Sharpe Ratio':
        sharpe_ratio = (returns - rf_rate / 100) / volatility
        return -sharpe_ratio

    elif metric == 'Sortino Ratio':
        downside_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252)
        sortino_ratio = (returns - rf_rate / 100) / downside_std
        return -sortino_ratio

# Optimization Function
def optimize_portfolio(mean_returns, cov_matrix, rf_rate, daily_returns, metric):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, rf_rate, daily_returns, metric)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    result = minimize(portfolio_performance, num_assets * [1. / num_assets],
                      args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# PDF Report Generator
def generate_pdf_report(opt_metric, annual_return, annual_volatility, VaR_95, optimized_weights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Portfolio Optimization Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Optimized for: {opt_metric}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Annualized Return: {annual_return:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Annualized Volatility: {annual_volatility:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"1-day 95% VaR: {VaR_95:.4f}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Optimized Weights:", ln=True)
    for ticker, weight in optimized_weights.items():
        pdf.cell(200, 10, txt=f"{ticker}: {weight:.2%}", ln=True)
    return pdf

# Run when optimize is clicked
if optimize_button:
    try:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
        prices = get_price_data(tickers, start_date, end_date)

        daily_returns = prices.pct_change().dropna()
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()

        result = optimize_portfolio(mean_returns, cov_matrix, rf_rate, daily_returns, opt_metric)
        optimized_weights = pd.Series(result.x, index=tickers)

        st.success(f"Portfolio optimized for **{opt_metric}** ✅")

        st.subheader("📊 Optimized Portfolio Allocation")
        fig1 = px.pie(values=optimized_weights, names=tickers, title="Asset Allocation")
        st.plotly_chart(fig1)

        st.subheader("📈 Historical Price Chart")
        fig2, ax = plt.subplots(figsize=(12, 5))
        prices.plot(ax=ax)
        st.pyplot(fig2)

        st.subheader("📉 Daily Returns Distribution")
        fig3, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(daily_returns, kde=True, ax=ax, bins=50)
        ax.set_title("Daily Returns Distribution")
        st.pyplot(fig3)

        annual_return = np.sum(mean_returns * result.x) * 252
        annual_volatility = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x))) * np.sqrt(252)

        st.subheader("📑 Portfolio Risk-Return Summary")
        st.write(pd.DataFrame({
            'Annualized Return': [f"{annual_return:.2%}"],
            'Annualized Volatility': [f"{annual_volatility:.2%}"],
            'Risk-Free Rate': [f"{rf_rate:.2f}%"]
        }))

        # Cumulative Returns Plot
        st.subheader("📈 Cumulative Returns")
        cumulative_returns = (1 + daily_returns).cumprod()
        fig4, ax = plt.subplots(figsize=(12, 5))
        cumulative_returns.plot(ax=ax)
        ax.set_title("Cumulative Returns Over Time")
        st.pyplot(fig4)

        # Correlation Heatmap
        st.subheader("🔥 Asset Returns Correlation Heatmap")
        fig5, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(daily_returns.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig5)

        # Value at Risk (VaR)
        st.subheader("📉 Portfolio Value at Risk (1-Day, 95%)")
        portfolio_daily_returns = daily_returns.dot(optimized_weights)
        VaR_95 = np.percentile(portfolio_daily_returns, 5)
        st.write(f"**1-day 95% Value at Risk (VaR): {VaR_95:.4f}**")

        # Download CSV
        csv = optimized_weights.to_csv().encode()
        st.download_button(
            label="📥 Download Allocation as CSV",
            data=csv,
            file_name='optimized_portfolio.csv',
            mime='text/csv',
        )

        # Download PDF Report
        pdf_report = generate_pdf_report(opt_metric, annual_return, annual_volatility, VaR_95, optimized_weights)
        pdf_bytes = pdf_report.output(dest='S').encode('latin1')
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="portfolio_report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

else:
    st.info("👈 Enter details and click 'Optimize Portfolio' to begin.")
