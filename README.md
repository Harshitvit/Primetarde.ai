# 🚀 CryptoTrader Sentiment Intelligence Dashboard

### 📊 Analyze the Impact of Market Sentiment on Trader Performance

This project explores how **Bitcoin market sentiment (Fear & Greed Index)** influences trader performance on **Hyperliquid historical trading data**.
It combines **data analysis, visualization, and machine learning** into an interactive **Streamlit dashboard**.



## 🧠 Project Overview

Financial markets are driven not only by fundamentals but also by **human emotions** — fear and greed.
This project aims to uncover how trader performance correlates with these sentiment shifts using real-world datasets.

**Objectives:**

* Explore the relationship between market sentiment and trading outcomes
* Identify behavioral trends among traders during Fear vs. Greed phases
* Build an interactive Streamlit dashboard for exploration and insights
* Provide statistical and visual evidence for sentiment-driven performance



## 📁 Datasets

### 1. 🪙 Bitcoin Market Sentiment Dataset

* **Columns:** `Date`, `Classification (Fear/Greed)`
* [Download from Google Drive](https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing)

### 2. 💹 Historical Trader Data (Hyperliquid)

* **Columns:** `account`, `symbol`, `execution_price`, `size`, `side`, `time`, `start_position`, `event`, `closedPnL`, `leverage`, etc.
* [Download from Google Drive](https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing)

---

## 🧰 Tech Stack

| Category                 | Tools                     |
| ------------------------ | ------------------------- |
| **Language**             | Python                    |
| **Dashboard Framework**  | Streamlit                 |
| **Visualization**        | Plotly, Matplotlib        |
| **Data Handling**        | Pandas, NumPy             |
| **Statistical Analysis** | Statsmodels, Scikit-learn |
| **Version Control**      | Git, GitHub               |

---

## ⚙️ Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/CryptoTrader-Sentiment-Dashboard.git
cd CryptoTrader-Sentiment-Dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install streamlit pandas numpy plotly statsmodels scikit-learn openpyxl
```

### 3. Run the app locally

```bash
streamlit run streamlit_das.py
```

### 4. Upload Datasets

In the sidebar:

* Upload the **Historical Trader Data**
* Upload the **Fear & Greed Index**
* Explore metrics, charts, and statistical insights interactively!

---

## 🌐 Streamlit Deployment

You can deploy this app easily on [Streamlit Cloud](https://streamlit.io/cloud).

**Steps:**

1. Push this project to a GitHub repository.
2. Go to Streamlit Cloud → “New app”.
3. Select your repo and `streamlit_das.py` file.
4. Add your Google Drive dataset links or upload CSVs.

🔗 **Live Demo:** https://primetardeai-bviw6ev96j2wuczvjyt7gq.streamlit.app/

---

## 📈 Key Features

✅ **Interactive Dashboard** — Filter by date, account, and sentiment
✅ **Daily Performance Aggregation** — PnL, volume, and win rate
✅ **Sentiment Impact Charts** — Fear vs. Greed performance comparison
✅ **Regression & Correlation Analysis** — Quantitative insight into effects
✅ **Data Export** — Download cleaned and merged datasets
✅ **Customizable Extensions** — Add cohort analysis, symbol insights, or backtests

---

## 🧩 Future Enhancements

* Integrate real-time sentiment API
* Add trading strategy backtesting
* Deploy on Docker or Streamlit Cloud
* Extend analysis to other assets (ETH, SOL, etc.)




