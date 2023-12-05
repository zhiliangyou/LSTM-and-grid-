# -*- coding: utf-8 -*-
"""
Created on Wen May 24 02:47:46 2023

@author: 35003
"""
import requests
import json
import time
import datetime
import hmac
import base64
import pandas as pd

'''para
'''
#http

API_URL = 'https://www.okx.com'
GET = "GET"
HISTORY_CANDLES = '/api/v5/market/history-candles'
VOLUMNE = '/api/v5/market/platform-24-volume'
TICKER_INFO = '/api/v5/market/ticker'
ORDER_BOOKS = '/api/v5/market/books'
SERVER_TIMESTAMP_URL = '/api/v5/public/time'
POST = "POST"
#header
APPLICATION_JSON = 'application/json'
CONTENT_TYPE = 'Content-Type'
OK_ACCESS_KEY = 'OK-ACCESS-KEY'
OK_ACCESS_SIGN = 'OK-ACCESS-SIGN'
OK_ACCESS_TIMESTAMP = 'OK-ACCESS-TIMESTAMP'
OK_ACCESS_PASSPHRASE = 'OK-ACCESS-PASSPHRASE'

##############################################################
##############################################################
'''
mod
'''
class base():
    def clean_dict_none(d: dict) -> dict:
        return {k:d[k] for k in d.keys() if d[k] != None}
# =============================================================================
#接收一个字典d作为输入，并返回一个新的字典。这个新的字典包含了原字典中所有非None值的键值对。
# 创建了一个新的字典。字典推导式的语法类似于列表推导式，但是它使用大括号{}而不是方括号[]，并且它需要一个键和一个值来创建每个字典项。
# 在这个字典推导式中，k是键，d[k]是值。for k in d.keys()部分遍历了原字典的所有键。if d[k] != None部分是一个条件，只有当值不等于None时，键值对才会被包含在新的字典中。
# 这个函数的作用就是创建一个新的字典，新字典中包含了原字典中所有非None值的键值对。
# =============================================================================
    def get_timestamp():
        now = datetime.datetime.utcnow()
        t = now.isoformat("T", "milliseconds")
        return t + "Z"
    
    def sign(message, secretKey):
        mac = hmac.new(bytes(secretKey, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
        d = mac.digest()
        return base64.b64encode(d)
    
    
    def pre_hash(timestamp, method, request_path, body):
        return str(timestamp) + str.upper(method) + request_path + body
    
    
    def get_header(api_key, sign, timestamp, passphrase,flag):
        header = dict()
        header[CONTENT_TYPE] = APPLICATION_JSON
        header[OK_ACCESS_KEY] = api_key
        header[OK_ACCESS_SIGN] = sign
        header[OK_ACCESS_TIMESTAMP] = str(timestamp)
        header[OK_ACCESS_PASSPHRASE] = passphrase
        header['FLAG'] = flag
        return header
    
    def parse_para_to_str(para):
        para = base.clean_dict_none(para)
        url = '?'
        for key, value in para.items():
            url = url + str(key) + '=' + str(value) + '&'
        return url[0:-1]

class Client(object):

    def __init__(self, api_key, api_secret_key, passphrase, use_server_time=False, flag='1'):

        self.API_KEY = api_key
        self.API_SECRET_KEY = api_secret_key
        self.PASSPHRASE = passphrase
        self.use_server_time = use_server_time
        self.flag = flag

    def _request(self, method, request_path, para):

        if method == GET:
            request_path = request_path + base.parse_para_to_str(para)
        # url
        url = API_URL + request_path

        timestamp = base.get_timestamp()

        # sign & header
        if self.use_server_time:
            timestamp = self.get_timestamp()

        body = json.dumps(para) if method == POST else ""

        sign = base.sign(base.pre_hash(timestamp, method, request_path, str(body)), self.API_SECRET_KEY)
        header = base.get_header(self.API_KEY, sign, timestamp, self.PASSPHRASE, self.flag)
        response = None
        if method == GET:
            response = requests.get(url, headers=header)
        elif method == POST:
            response = requests.post(url, data=body, headers=header)
        if not str(response.status_code).startswith('2'):
            raise exceptions.OkexAPIException(response)

        return response.json()


    def request_with_para(self, method, request_path, para):
        return self._request(method, request_path, para)

    def get_timestamp(self):
        url = API_URL + SERVER_TIMESTAMP_URL
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['ts']
        else:
            return ""


    
class MarketAPI(Client):

    def __init__(self, api_key, api_secret_key, passphrase, use_server_time=False, flag='1'):
        Client.__init__(self, api_key, api_secret_key, passphrase, use_server_time, flag)

    #Get Candlesticks History
    def get_history_candlesticks(self, instId, after=None, before=None, bar=None, limit=None):
        para = {'instId': instId, 'after': after, 'before': before, 'bar': bar, 'limit': limit}
        return self.request_with_para(GET, HISTORY_CANDLES, para)


################################################################################
'''
get data
'''

if __name__ == '__main__':
    api_key = ""
    secret_key = ""
    passphrase = ""   
    flag = '1' 
    marketAPI = MarketAPI(api_key, secret_key, passphrase, False,flag)
    
    result = marketAPI.get_history_candlesticks('BTC-USDT')
    df = pd.DataFrame(result['data'])
    new_column_names=['date','open','high','low','close']
    df.columns = new_column_names + list(df.columns[5:])
    df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], unit='ms')
    print(df.head(10))



















