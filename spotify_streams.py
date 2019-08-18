import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime, timedelta
from dateutil.relativedelta import *

# Extract a list of dates 30 days from today
list_date = []

for i in range(30):
    list_date.append(datetime.now().date() + relativedelta(days=-i))

# Retrieve info from Spotify url (30 days) and convert it to DataFrame
spotify_data = pd.DataFrame()

for i in range(2,30):
    url = 'https://spotifycharts.com/regional/us/daily/' + str(list_date[i]) + '/download'
    r = requests.get(url)
    s = r.text
### Delete the first row in s (not part of the data)
    delete = s.split('\n')[0]
    delete = delete + '\n'
    s = s.lstrip(delete)
### Convert to DataFrame
    t = StringIO(s)
    data = pd.read_csv(t)
    data['Date'] = list_date[i]
    spotify_data = spotify_data.append(data, ignore_index=True)

def get_data_track(track, artist):
    interest = spotify_data.loc[(spotify_data['Track Name']==track) & (spotify_data['Artist']==artist)][['Date','Position','Streams']]
    plt.plot(interest['Date'], interest['Streams'])
    plt.xlabel('Date')
    plt.ylabel('Streams')
    plt.xticks(rotation=60)
    plt.title(track + ' by ' + artist)
    plt.tight_layout()
    plt.style.use('fivethirtyeight')
    plt.show()

def get_data_artist(artist):
    interest = spotify_data.loc[spotify_data['Artist']==artist][['Date','Position','Streams']]
    interest1 = interest.groupby('Date', as_index=False)['Streams'].sum()
    plt.plot(interest1['Date'], interest1['Streams'])
    plt.xlabel('Date')
    plt.ylabel('Streams')
    plt.xticks(rotation=60)
    plt.title(artist)
    plt.tight_layout()
    plt.style.use('fivethirtyeight')
    plt.show()