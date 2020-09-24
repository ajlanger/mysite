from django.db import models
from django import forms
# from .crypto_bot.fi import get_all_possible_coins, get_coin_data

# Create your models here.
coins = ['ADA', 'ATOM', 'BCH', 'DASH', 'EOS', 'GNO', 'QTUM', 'XETCZ', 'XETHZ', 'XLTCZ', 'XREPZ', 'XTZ', 'XXBTZ', 'XXLMZ', 'XXMRZ', 'XXRPZ', 'XZECZ']

labels = ['ADA', 'ATOM', 'BCH', 'DASH', 'EOS', 'GNO', 'QTUM', 'ETC', 'ETH', 'LTC', 'REP', 'XTZ', 'XBT', 'XLM', 'XMR', 'XRP', 'ZEC']

ALL_POSSIBLE_COINS = [(c, l) for c, l in zip(coins, labels)]
POSSIBLE_INTERVALS = [('daily', 'Daily'), ('hourly', 'Hourly')]

class cryptoData(forms.Form):
    possible_coins = ALL_POSSIBLE_COINS
    interval = forms.CharField(label='Interval', widget=forms.Select(choices=POSSIBLE_INTERVALS))
    coinchoice = forms.CharField(label='Crypto', widget=forms.Select(choices=ALL_POSSIBLE_COINS))
