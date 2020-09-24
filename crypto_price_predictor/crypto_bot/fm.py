# Deciders_Executables
# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Libraries
import supports
import fed
import fc, fi
import matplotlib.pyplot as plt
import datetime
import requests
import sys
import importlib
import evaluator
importlib.reload(evaluator)


# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Functions and classes
def initiate_bot(trade_criteria, redundant_argument):
    """""""""
    FM1 Function to initiate script
    (input: ...)
    (output: NOTHING)
    """""""""
    euros_to_spend = trade_criteria['euros_to_spend']
    amount_of_coins_to_trade = trade_criteria['amount_of_coins_to_trade']

    # First check for open buy orders, I don't want to buy too much, selling isn't a problem
    go = True
    open_orders = fi.get_open_orders()['open']
    for order in open_orders:
        if open_orders[order]['descr']['type'] == 'buy':
            go = False

    portfolio_data = fi.get_processed_portfolio(euros_to_spend)
    transaction_history = fi.get_transaction_history()

    # ------------------------------- SELLING COMPONENT ------------------------------------------------------------
    print('------------ SELLING COMPONENT -------------')

    if len(portfolio_data[0]) != 0:
        print(portfolio_data[0])
        for coin in portfolio_data[0]:
            """""""""
            nieuw gedeelte: op dit moment kijkt de bot naar de laatste buy transactie van de coin, en kijkt ze naar die
            prijs om de kost te berekenen. Wat ik wil is dat ze alle buy transacties neemt van die ene coin, tot ze
            aan het aantal komt dat nu in men portfolio zitten, en telkens het aantal * de aankoopprijs doet om de
            totale kost te weten, en een sell/hold beslissing neemt. Wanneer het een sell beslissing is, moet ze alle
            coins verkopen
            """""""""
            coin_amount = 0
            total_buy_cost = 0
            i = 0
            while str(round(coin_amount, 4))[:3] != str(round(portfolio_data[0][coin], 4))[:3]:
                if coin in transaction_history[i]['coin'] and transaction_history[i]['action'] == 'buy':
                    coin_amount += float(transaction_history[i]['amount'])
                    total_buy_cost += float(transaction_history[i]['total'])
                i += 1

            """""""""
            Hoe vroeger de prijzen werden vergeleken (kijken naar de allerlaatste buy actie van de coin):

            latest_crypto_buy = fi.get_latest_crypto_transaction(coin, transaction_history, 'buy')
            current_position_value = float(latest_crypto_buy['amount']) * float(fi.get_current_currency_price(coin)['bid'])
            print(f'coin: {coin}, \ncost: {latest_crypto_buy["amount"]} coins * €{float(latest_crypto_buy["total"])/float(latest_crypto_buy["amount"])} = {float(latest_crypto_buy["total"])}\n'
                  f'current value now: {current_position_value}')
            """""""""

            current_position_value = float(portfolio_data[0][coin]) * float(fi.get_current_currency_price(coin)['bid'])
            print(f'coin: {coin}, \n'
                  f'total cost: {total_buy_cost}\n'
                  f'current value now: {current_position_value}')

            if float(total_buy_cost) < current_position_value:
                print(f'Sell {coin} (total buy cost: {total_buy_cost}, '
                      f'total value now: {current_position_value})')
                # print(fed.place_order('sell', coin, portfolio_data[0][coin], fi.get_current_currency_price(coin)['bid'])['result'])
            print('---------------------------------------------------------------------------------------------------')

    if go:
        # -------------------------------- BUYING COMPONENT ------------------------------------------------------------
        if len(portfolio_data[0]) < trade_criteria['amount_of_coins_to_trade']:
            print('------------ BUYING COMPONENT -------------')
            print('portfolio data: ', portfolio_data)
            euros_to_spend_per_coin = euros_to_spend / amount_of_coins_to_trade

            # Coin kopen als slope (over 15 dagen) positief is en ze op 1 dag gedaald is in prijs

            # Dus eerst zoeken naar coins die gedaald zijn in prijs op 1 dag
            buy_dict_2 = fi.get_info_sell_or_buy(intelligence_level=0)['buy']

            # Zet dictionary op zo'n manier dat de coin die het meest daalde eerst staat
            sorted_buy_dict_2 = {k: buy_dict_2[k] for k in sorted(buy_dict_2.keys(), key=buy_dict_2.__getitem__)}
            buy_coins_list = list(sorted_buy_dict_2)

            # Ik wil niet dat de bot coins koopt die ik al in bezit heb
            for coin_in_portfolio in portfolio_data[0]:
                for coin_to_buy in buy_coins_list:
                    if coin_in_portfolio == coin_to_buy:
                        buy_coins_list.remove(coin_to_buy)

            if buy_dict_2 != {} and buy_coins_list != []:
                buy_transaction_done = False

                for buy_times in range(0, len(buy_coins_list)):
                    if buy_times != len(buy_coins_list):
                        if portfolio_data[1]['Euros available'] < euros_to_spend_per_coin:
                            return f'Not enough euros available (euros available: {portfolio_data[1]["Euros available"]})'

                        price = fi.get_current_currency_price(buy_coins_list[buy_times])
                        amount = euros_to_spend_per_coin / float(price['ask'])

                        print(f'------------------------------------------- \n'
                              f'{price["coin"]}')

                        # Check whether the amount is equal to or higher than the minimum buy amount
                        minimum_amounts = fi.get_minimum_order_amount()
                        for coin in minimum_amounts:
                            if coin in price['coin']:
                                if amount < minimum_amounts[coin]:
                                    print(f'Adjust amount of {amount} to {minimum_amounts[coin]}')
                                    amount = minimum_amounts[coin]

                        # --------------------------------------------------- Berekenen van slope!
                        buy_dict_1 = fi.get_coin_data(buy_coins_list[buy_times])
                        # Make sure to import the necessary packages and modules
                        # Preparing the data
                        datum_list = []
                        price_list = []

                        for value in buy_dict_1[buy_coins_list[buy_times]][::-1]:
                            if '2019' in value[0]:
                                datum_list.append(datetime.datetime.strptime(value[0], '%Y-%m-%d').date())
                                price_list.append(float(value[1]))

                        x_day_prices = price_list[:6]
                        x_days = datum_list[:6]

                        slope_x_days = x_day_prices[0] - x_day_prices[-1]
                        """""""""
                        # Plotting of graph
                        plt.plot(x_days, x_day_prices, label=f'prices {buy_coins_list[buy_times]}')
                        plt.plot([x_days[0], x_days[-1]], [x_day_prices[0], x_day_prices[-1]],
                                 label='slope')

                        # Showing the result
                        plt.show()
                        """""""""
                        print(f'Check of {price["coin"]}, slope over 15 days: {slope_x_days}')
                        euros_to_spend_per_coin = float(amount) * float(price['ask'])

                        if slope_x_days >= 0:
                            buy_transaction_done = True
                            print(f'Buy {amount} €{price["coin"]}s at €{price["ask"]} per coin with €{euros_to_spend_per_coin}.')
                            # print(fed.place_order('buy', buy_coins_list[buy_times], amount, price['ask'])['result'])

                if buy_transaction_done:
                    return 'Coin(s) were bought'
                elif not buy_transaction_done:
                    return 'The buy_coins_list was rendered empty, there are no interesting coins.'
            print('---------------------------------------------------------------------------------------------------')

    return f'There were still open buy orders, or the max amount of coins ({trade_criteria["amount_of_coins_to_trade"]}) was reached. Current amount of coins={len(portfolio_data[0])}.'


def initiate_bot_v2(trade_criteria, redundant_argument):
    # Buy
    all_coins   = fi.get_all_possible_coins()
    buy_advices = evaluator.get_advices(all_coins, number=5, min_acc=0.6, interval='hourly', advice_kind='b')

    # Sell
    possible_coins = list(fi.get_processed_portfolio(111)[0].keys())
    if possible_coins == []:
        sell_advices = 'No crypto available to sell'
    else:
        sell_advices = evaluator.get_advices(possible_coins, number=5, min_acc=0.6, interval='hourly', advice_kind='s')

    return {'buy_advices':buy_advices, 'sell_advices':sell_advices}


# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

#print(initiate_bot({'euros_to_spend': 50, 'amount_of_coins_to_trade': 5}, 'something'))  # Laat mij 10€ spenderen per coin

# test = {'ADA': -0.007112375533428141, 'ATOM': -0.025746921159355556, 'XETCZ': -0.0017496111975117304, 'XXRPZ': -0.0013689521649001327}
