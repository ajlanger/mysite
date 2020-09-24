# Libraries
from .supports import dates_by_days_prior
import math
import datetime
import requests
from time import gmtime, strftime
import sys
import seaborn as sns
import pandas as pd
sns.set(style="darkgrid")

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# FI1
def get_posts(site):
    """""""""
    FI1 Function to search for all the posts about certain crypto currencies and have advices to buy/sell
    (input: authenticated_login)
    (output: lists_with_coin_names_and_words_or_sentences_that_indicate_buy_or_sell_and_when)
    example: [[ETH, "Good", "WTF that's bad", "should buy"][BTC, "bad", "bad"][...]]
    """""""""
    return site


# FI2
def get_info_sell_or_buy(intelligence_level=0):
    """""""""
    FI2 Function to search for information about what to sell/buy
    (input: lists_with_coin_names_and_words_or_sentences_that_indicate_buy_or_sell_and_when)
    (output: library_with_coin_suggestion_and_date)
    example: {ETH: {action: buy, date: 01/08/2019}, BTC: {action: sell, date:01/08/2019}}
    """""""""
    buy = {}
    sell = {}
    actual_output = {}
    if intelligence_level == 0:
        for coin in get_all_possible_coins():
            # If previous days == 1: the difference is made between the price of yesterday and the price today
            previous_days = 1
            price_average = get_price_average(coin=coin, previous_days=previous_days)
            today = str(datetime.datetime.now())[:10]
            price_now = get_coin_data(coin=coin, optional_date=today)
            price_now = float(price_now[coin][1])
            #print(f'{coin} Price now: {price_now}, price average: {price_average}')

            if previous_days == 1:
                message1 = 'less than yesterday'
                message2 = 'more than yesterday'
            else:
                message1 = f'less than the average of {previous_days} days'
                message2 = f'more than the average of {previous_days} days'

            if price_now < price_average:
                try:
                    buy[coin] = (price_now - price_average)/price_average
                except:
                    buy[coin] = price_now
            else:
                try:
                    sell[coin] = (price_now - price_average)/price_average
                except:
                    sell[coin] = price_now

        actual_output['buy'] = buy
        actual_output['sell'] = sell
        return actual_output

# FI3: OK
def get_coin_data(coin='EOS', optional_date='None', interval='daily'):
    """""""""
    FI3 Function about how much a coin is/was worth on a given date
    (input: coin, date_time=time.today())
    (output: price_on_given_date)
    example: {BTC: 1596€}
    """""""""
    coinpair = f'{coin}eur'.lower()

    interval_minutes = {'weekly':10080, 'daily':1440, 'hourly':60, 'half-hourly':30, 'minutely':1}

    output = requests.get('https://api.kraken.com/0/public/OHLC', params={'pair': coinpair, 'interval': interval_minutes[interval]}).json()
    output_list = []
    dict_keys = output['result'].keys()
    dict_keys = list(dict_keys)
    for coindata in output['result'][dict_keys[0]]:
        # Append hours and prices + Correct time if needed
        datestring = datetime.datetime.utcfromtimestamp(coindata[0]).strftime('%d-%m-%y %H:%M')
        date = datetime.datetime.strptime(datestring, '%d-%m-%y %H:%M') + datetime.timedelta(hours=2)

        date = date.strftime('%d-%m-%y %H:%M')

        output_list.append([date, float(coindata[1])])

    if optional_date == 'None':
        return {coin: output_list}
    else:
        for date in output_list:
            if date[0] == optional_date:
                return {coin: date}


def get_datetime_object(date_string):
    return datetime.datetime.strptime(date_string, '%d-%m-%y %H:%M')


def get_time_series_plot(coin='eos', interval='daily'):
    coin_data = get_coin_data(coin=coin, interval=interval)
    time_dict = [get_datetime_object(el[0]) for el in coin_data[list(coin_data)[0]]]
    coin_dict = [price[1] for price in coin_data[list(coin_data)[0]]]
    # Normalize the price list
    norm = [float(i)/max(coin_dict) for i in coin_dict]

    d = {'time':time_dict, 'price':coin_dict}
    pricedata = pd.DataFrame(data=d)
    return sns.lineplot(x='time', y='price',data=pricedata,
                  markers=True, dashes=False)


# FI6
def get_price_average(coin='EOS', previous_days=31):
    """""""""
    FI6 Function about what the average is of a coin for a certain amount of days
    (input: coin, days)
    (output: price_average)
    example: 230€
    """""""""
    # Begin calculating the average
    coin_values = get_coin_data(coin)[coin]
    output_values = []
    for day in dates_by_days_prior(previous_days):
        for date in coin_values:
            if date[0] == day:
                output_values.append(date[1])

    price_sum = 0
    for value in output_values:
        price_sum += float(value)
    price_average = price_sum / previous_days
    return price_average


# FIX: Get present ask/bud prices
def get_current_currency_price(coin):
    # Make sure the crypto is written correctly
    for crypto in get_all_possible_coins():
        if coin in crypto:
            coin = crypto
    coinpair = f'{coin.lower()}eur'
    parameters = {'pair': coinpair}
    asset = requests.get('https://api.kraken.com/0/public/Ticker', params=parameters)
    asset_lib = asset.json()
    return {'date': strftime("%Y-%m-%d %H:%M:%S", gmtime()).split(' ')[0],
            'time': strftime("%Y-%m-%d %H:%M:%S", gmtime()).split(' ')[1],
            'coin': coin,
            'ask': asset_lib['result'][coinpair.upper()]['a'][0],
            'bid': asset_lib['result'][coinpair.upper()]['b'][0]}


# FI11
def get_all_possible_coins():
    output = requests.get('https://api.kraken.com/0/public/AssetPairs').json()
    coin_list = []
    for coin in output['result']:
        if 'EUR' in coin:
            if str(coin)[:coin.find('EUR')] != '' and str(coin)[:coin.find('EUR')] != 'Z':
                coin_list.append(str(coin)[:coin.find('EUR')])
    return remove_duplicates(coin_list)

# Remove duplicate strings in a list
def remove_duplicates(list):
    output_list = []
    for el in list:
        if el not in output_list:
            output_list.append(el)
    return output_list


# FI12
def compare_opening_prices(coin, date):
    for data in get_coin_data(coin=coin)[coin]:
        if data[0] == date:
            coinpair = f'{coin}eur'.lower()
            output = requests.get('https://api.kraken.com/0/public/Ticker', params={'pair': coinpair}).json()
            output_dict = {'coin': coin,
                           'today opening price': output['result'][coinpair.upper()]['o']
                           }
            return {'coin': coin,
                    'today opening price': output_dict['today opening price'],
                    'opening price prior': data}


# FI13
def get_processed_portfolio(euro_balance):
    """""""""
    get the list of actual coins I can trade, and also my available euros seperately
    """""""""
    # Look if I have any coins, make a dict of all available coins (cryptos) and the euros I want to trade with
    euros = {'Euros to trade with': euro_balance}
    coins_available = {}
    portfolio = get_current_portfolio()
    for coin in portfolio:
        if float(portfolio[coin]) > 0.001 and 'EUR' not in coin:
            coins_available[coin] = float(portfolio[coin])
        elif 'EUR' in coin:
            euros['Euros available'] = float(portfolio[coin])

    return coins_available, euros


def get_latest_crypto_transaction(crypto, transaction_list, action):
    latest_transaction_time = datetime.datetime.strptime('1970-01-01 19:30', '%Y-%m-%d %H:%M')
    latest_transaction_data = {}
    for transaction in transaction_list:
        if crypto in transaction['coin'] and transaction['action'] == action:
            if datetime.datetime.strptime(transaction['time'], '%Y-%m-%d %H:%M') > latest_transaction_time:
                latest_transaction_time = datetime.datetime.strptime(transaction['time'], '%Y-%m-%d %H:%M')
                latest_transaction_data = transaction
    return latest_transaction_data

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

"""""""""""
api_key = 'a4d9b441500b6ec3f8e9601e4a1c5cecbf1d6fa784477de3af00c8801af4581a'
response = requests.get(url='https://min-api.cryptocompare.com/data/price?fsym=EUR&tsyms=[BTC,EOS]', headers={'apikey': api_key})

print(response.json())
"""""""""""

def get_minimum_order_amount():
    output = str(requests.get('https://support.kraken.com/hc/en-us/articles/205893708-What-is-the-minimum-order-size-').content)
    output_list = output.split('<')
    actual_output_list = []
    for i in output_list:
        if 'td class="wysiwyg-text-align-right">' in i:
            actual_output_list.append(i[i.find('>')+1:])
    actual_output_library = {}
    for minimal_amount in actual_output_list[1:]:
        actual_output_library[minimal_amount.split(' ')[1]] = float(minimal_amount.split(' ')[0])

    return actual_output_library

"""""""""
output = str(requests.get('https://support.kraken.com/hc/en-us/articles/205893708-What-is-the-minimum-order-size-').content)
print(output)
"""""""""
