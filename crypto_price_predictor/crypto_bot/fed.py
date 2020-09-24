# Deciders_Executables
# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Libraries
import fc
import fi
# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Functions and classes

"""""""""
                FD1 Function to determine highest profit
                (input: library_with_possible_actions)
                IF Buy: What is the highest predicted value of a certain coin?
                IF Sell: What sell will yield the highest profit?
                Make comparison between future value long and current value of short
                (output: library_with_ONE_action)
                example: {ETH: {action: Sell, amount: 3, date_time: 22/08/2019 23h, price: 200€}}
"""""""""


def place_order(type, coin, amount, price):
    """""""""
    FE1 Function to place the buy or sell order of a certain coin
    (input: coin, date_time, price, buy_or_sell)
    (output: successful_or_not)
    """""""""
    for crypto in fi.get_all_possible_coins():
        if coin in crypto:
            coin = crypto

    coinpair = f'{coin}eur'.lower()

    return fc.SiteLogger().twitter_reddit_kraken_coinapi(site='kraken', argument='AddOrder', argument2={'pair': coinpair,
                         'type': type,
                         'ordertype': 'limit',
                         'price': price,
                         'volume': amount})

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

