import numpy as np
import pandas as pd
import spacy
import streamlit as st

#python -m spacy download en

st.title('Recommendation system')
st.subheader('The recommendation system identifies similar items based on multiple approaches')

file = 'data.csv'
df_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
       'UnitPrice', 'CustomerID', 'Country']
df = pd.read_csv(file, usecols=df_columns, encoding='latin1')
#nlp = spacy.load("en_core_web_sm")

#data cleaning
df['InvoiceDay'] = pd.to_datetime(df['InvoiceDate']).dt.to_period('D') # convert all the dates to year-month-day format
df = df[~df['Quantity'] < 0] # remove refunds etc

#one description per stock code
df_description = df.groupby(['StockCode'])['Description'].apply(pd.Series.mode)
df = pd.merge(df, df_description, how='left', on='StockCode')
df = df.drop('Description_x', axis=1)
df = df.rename(columns={'Description_y': 'Description'})

#encode customerID as string
df = df.astype({'CustomerID': object}) 

# drop strange items
df = df[~df['StockCode'].str.contains('gift')] #drop gift vouchers
df = df[~df['StockCode'].str.contains('BANK')] #drop bank charges
df = df[~df['StockCode'].isin(['AMAZONFEE', 'S', 'M', 'm', 'B'])] #sample
df = df[~df['Quantity'] < 0] #drop returns, it is not clear whether refunded items can be added back to the inventory

# preprocessing to homogenize the descriptions
df['Description'] = df['Description'].str.replace('.', '')
df['Description'] = df['Description'].str.replace('/', ' OF ')
df['Description'] = df['Description'].str.replace(r'\s{2,}', ' ')
df = df[df['Description'] != '']

# remove items that are only purchased once
item_by_popularity = df['Description'].value_counts()
purchased_only_once = item_by_popularity[item_by_popularity == 1].index
df = df[~df['Description'].isin(purchased_only_once)]

common_items = df['Description'].value_counts().keys()[:15]
methods = ["content-based", "collaborative"]

df_unique = pd.DataFrame(df['Description'].unique(), columns=['Description'])

item = st.sidebar.selectbox("Select an item for a recommendation.", tuple(common_items))
method = st.sidebar.selectbox("Select a method.", tuple(methods))

def content_based(item):
    doc = nlp(item)
    nouns = []
    proper_nouns = [] #backup
    for token in doc:
    pos = token.pos_
    if pos == 'NOUN':
      nouns.append(token.text)
    elif pos == 'PROPN':
      proper_nouns.append(token.text)
    else:
      pass
    if len(nouns) == 0:
    nouns += proper_nouns

    keyword = ''.join([f'{noun}|' for noun in nouns])[:-1]
    is_match = df_unique['Description'].str.contains(keyword, na=False)
    results = df_unique[is_match].iloc[1:6]
    return results

def collaborative(item):
    return item

st.write(f'Great choice! Here are our recommendations based on {item}')
if method == 'content-based':
    chosen_items = content_based(item)
else:
    chosen_items = collaborative(item)
st.write(item)