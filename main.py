from flask import Flask, render_template,request
import pandas as pd
from App.application import next_seven_days_stock_forecast, get_stock_sentiment, get_stock_data,get_today_live_stock_data
from flask_bootstrap import Bootstrap 
import warnings
warnings.filterwarnings("ignore")
from streamlit import caching



app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/insertintotable',methods = ['POST'])
def insertintotable():
    caching.clear_cache()
    nm = request.form.get("nm")
    stock_symbol = nm
    get_stock_data(stock_symbol)
    df = pd.read_csv(''+stock_symbol+'.csv')
    today_stock = df.tail(1)
    df = df[["Close"]]

    if len(df) == 0:
    	return render_template('index.html',not_found=True , symbol = stock_symbol)
    else:
        get_today_live_stock_data(stock_symbol)
        p_values = [2, 4, 6]
        d_values = [1]
        q_values = range(0, 3)
        next_seven_days_forecast = next_seven_days_stock_forecast(stock_symbol, df, p_values, d_values, q_values)
        suggestion, tw_list, tw_pol = get_stock_sentiment(stock_symbol)
        
  
        
    return render_template('results.html',quote = stock_symbol,open_s = today_stock['Open'].to_string(index=False),
                               close_s=today_stock['Close'].to_string(index=False),adj_close=today_stock['Adj Close'].to_string(index=False),
                               tw_list=tw_list,tw_pol=tw_pol,decision=suggestion,high_s=today_stock['High'].to_string(index=False),
                               low_s=today_stock['Low'].to_string(index=False),vol=today_stock['Volume'].to_string(index=False),
                               forecast_set=next_seven_days_forecast, arima_pred = next_seven_days_forecast[0])


if __name__ == '__main__':
   app.run()




