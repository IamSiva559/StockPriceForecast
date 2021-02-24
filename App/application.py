import pandas as pd
import numpy as np
from pandas import datetime
import yfinance as yf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import tweepy
from preprocessor.api import clean
import re
from textblob import TextBlob
import matplotlib.pyplot as plt

#Constants

CONSUMER_KEY= 'nlxa0UiERGBRU3xdnYDki0zXF'
CUSTOMER_SECRET= 'mRCTBVRZyZvFDZVlHypZQe5O6tZZpCeseRCIxbEska685sqoI4'

ACCESS_TOKEN='759618066-EPW2DXdStQO8Y5kxm87nZ8w1YRyG5XsyJOnNXLKd'
ACCESS_TOKEN_TOKEN='NUCBdTuERoFzf2vqgXEltBtqWRgNeIGGYpjbCb8Ztn4ee'

NUM_OF_TWEETS = int(300)



def get_stock_data(stock_symbol):
    end = datetime.now()
    start = datetime(end.year-2,end.month,end.day)
    data = yf.download(stock_symbol, start=start, end=end)
    df = pd.DataFrame(data=data)
    df.to_csv(''+stock_symbol+'.csv')
    return 

def get_today_live_stock_data(stock_symbol):
	live_stock_data = yf.download(tickers = stock_symbol , period = "1d", interval="5m")
	fig = live_stock_data.Close.plot()
	plt.savefig("static/"+stock_symbol+"_livedata"+'.png')
	return

def evaluate_arima_model(df, arima_order):
    # prepare training dataset
    train_size = int(len(df) * 0.99)
    train, test = df[0:train_size], df[train_size:]
    history = [x for x in train.Close]
    test = list(test.Close)
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0][0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return rmse

def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_order = rmse, order
                except:
                       continue
    #print('Best ARIMA%s MSE=%.3f' % (best_order, best_score))
    return best_order


def next_seven_days_stock_forecast(stock_symbol, df, p_values, d_values, q_values):
    best_arima_order = evaluate_models(df, p_values, d_values, q_values)
    # ARIMA Model
    model = ARIMA(df.Close, best_arima_order) 
    result = model.fit(disp=0)
    predictions = result.predict(start = len(df)-5, end = len(df)+7, typ='levels')
    predictions = np.array(predictions)
    next_seven_days_forcast = [predictions[i] for i in range(5, 12)]
    fig = result.plot_predict(start= len(df)-50, end= len(df)+10, dynamic=False)
    fig.savefig("static/"+stock_symbol+"_arima"+'.png')

    return next_seven_days_forcast



  
def get_stock_sentiment(stock_symbol):
    symbol = stock_symbol.split(".")[0].lower() + " stock"
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CUSTOMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_TOKEN)
    user = tweepy.API(auth)
    tweets = tweepy.Cursor(user.search, q=str(symbol), tweet_mode='extended', lang='en',exclude_replies=True).items(NUM_OF_TWEETS)
    tweet_list = [] #List of tweets alongside polarity
    global_polarity = 0 #Polarity of all tweets === Sum of polarities of individual tweets
    tw_list=[] #List of tweets only => to be displayed on web page
    #Count Positive, Negative to plot pie chart
    pos=0 #Num of pos tweets
    neg=1 #Num of negative tweets
    for tweet in tweets:
        count=20 #Num of tweets to be displayed on web page
        #Convert to Textblob format for assigning polarity
        tw2 = tweet.full_text
        tw = tweet.full_text
        #Clean
        tw=clean(tw)
        #print("-------------------------------CLEANED TWEET-----------------------------")
        #print(tw)
        #Replace &amp; by &
        tw=re.sub('&amp;','&',tw)
        #Remove :
        tw=re.sub(':','',tw)
        #print("-------------------------------TWEET AFTER REGEX MATCHING-----------------------------")
        #print(tw)
        #Remove Emojis and Hindi Characters
        tw=tw.encode('ascii', 'ignore').decode('ascii')

        #print("-------------------------------TWEET AFTER REMOVING NON ASCII CHARS-----------------------------")
        #print(tw)
        blob = TextBlob(tw)
        polarity = 0 #Polarity of single individual tweet
        for sentence in blob.sentences:
            polarity += sentence.sentiment.polarity
            if polarity>0:
                pos=pos+1
            if polarity<0:
                neg=neg+1

            global_polarity += sentence.sentiment.polarity
        if count > 0:
            tw_list.append(tw2)

        #tweet_list.append(Tweet(tw, polarity))
        count=count-1
    try:
    	global_polarity = global_polarity / len(tw_list)
    except:
    	pass

    neutral=NUM_OF_TWEETS-pos-neg
    if neutral<0:
        neg=neg+neutral
        neutral=20
    labels=['Positive','Negative','Neutral']
    sizes = [pos,neg,neutral]
    explode = (0, 0, 0)
    fig = plt.figure(figsize=(7.2,4.8),dpi=65)
    fig1, ax1 = plt.subplots(figsize=(7.2,4.8),dpi=65)
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.savefig("static/"+stock_symbol+"_sa"+'.png')
    plt.close(fig)
    #plt.show()
    if pos-neg > 10:
    	suggestion = " According to sentiment analysis, stock price may RISE"
    	tw_pol = " Overall Positive"
        
    elif neg-pos > 10:
        suggestion = " According to sentiment analysis, stock price may FALL"
        tw_pol = " Overall Negative"

    else:
        suggestion = " According to sentiment analysis, there wont be much volatilty in the stock price"
        tw_pol = " Almost neautral "


    return suggestion, tw_list, tw_pol
