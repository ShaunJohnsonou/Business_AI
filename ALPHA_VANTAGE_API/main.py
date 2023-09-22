import os
import requests
from pprint import pprint
import pymongo
import sys
import time
current_wd = os.getcwd()
client = pymongo.MongoClient('localhost', 27017)
db = client['Business_AI']
FUNDAMENTAL_DATA = ['COMPANY_OVERVIEW', 'INCOME_STATEMENT', 'CASH_FLOW', 'EARNINGS', 'LISTING_STATUS', 'EARNINGS_CALENDAR', 'IPO_CALENDAR']
def alpha_vantage_api(function, symbol, collection, api):
    global db
    Collection = db[collection]
    existing_document = Collection.find_one(symbol)

    if existing_document is None:
        url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api}'
        r = requests.get(url)
        data = r.json()
        data["_id"] = symbol
        data['function'] = function
        result = Collection.insert_one(data)
    else:
        print('The companys data is already in the database')
    




symbol = 'NPN'
with open(f'{current_wd}\\ALPHA_VANTAGE_API\\api_keys.txt', 'r') as api:
    api = api.read()
for func in FUNDAMENTAL_DATA:
    try:
        print(func)
        alpha_vantage_api(function=func, symbol=symbol, collection=func, api=api)   
    except Exception as e:
        print(e)
        # sys.exit(1)
    # time.sleep(5)
    