# Extractors
# Connector
# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Libraries
import requests
import krakenex
# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# Credentials
# CoinAPI
coinapi_headers = {'X-CoinAPI-Key': '16CAF496-C1BE-461B-8434-0A6B238A74F4'}

# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# Functions and classes
# FC1/2/3/4: NOK
class SiteLogger():
    def twitter_reddit_kraken_coinapi(self, site='', url='', argument='Balance',
                                      argument2={}, headers='head', parameters=''):
        if site == 'kraken':
            k = krakenex.API()
            k.load_key('kraken.key')
            if argument2 == {}:
                output = k.query_private(argument)
            else:
                output = k.query_private(argument, argument2)
            return output

        elif site == 'coinapi':
            headers = coinapi_headers
            output = requests.get(url, headers=headers)
            return output.json()

        elif site == 'reddit':
            return 0

        elif site == 'twitter':
            return 0

k = krakenex.API()
k.load_key('kraken.key')
k.query_private('TradeBalance', data= {'asset':'ZEUR'})

SiteLogger().twitter_reddit_kraken_coinapi(site='kraken', argument='TradeBalance', argument2={'asset':'ZEUR'})
# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
