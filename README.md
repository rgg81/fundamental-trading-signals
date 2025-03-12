# 📈 Fundamental Trading Signals

[![Build Status](https://github.com/rgg81/fundamental-trading-signals/actions/workflows/ci.yml/badge.svg)](https://github.com/rgg81/fundamental-trading-signals/actions)

An open-source project for generating **buy/sell signals** based on **fundamental macroeconomic data** like **inflation, interest rates, money supply, and trade balances**.

## 🚀 Features
✅ Fetch **fundamental macroeconomic indicators** from sources like **FRED**  
✅ Preprocess and transform raw economic data into **usable features**  
✅ Apply **machine learning models** (Regime-Switching, Random Forests, LSTMs)  
✅ Generate **buy/sell trading signals** based on macroeconomic trends  
✅ Backtest strategies with **historical data**  

## 📊 Data Sources
We fetch macroeconomic indicators from:
- **[FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/)**
- ECB, World Bank (planned)

## 🛠 Installation
```bash
git clone https://github.com/rgg81/fundamental-trading-signals.git
cd fundamental-trading-signals
pip install -r requirements.txt
