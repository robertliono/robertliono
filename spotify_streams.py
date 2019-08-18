import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime, timedelta
from dateutil.relativedelta import *

# Extract a list of dates 730 days from today (2 years)
list_date = []

for i in range(730):
    list_date.append(datetime.now().date() + relativedelta(days=-i))
	
print(list_date)

# Retrieve info from Spotify url (730 days) and convert it to DataFrame
spotify_data = pd.DataFrame()

for i in range(2,730):
    url = 'https://spotifycharts.com/regional/global/daily/' + str(list_date[i]) + '/download'
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
    '''This is to plot the number of streams vs date for any particular track.'''
    interest = spotify_data.loc[(spotify_data['Track Name']==track) & (spotify_data['Artist']==artist)][['Date','Position','Streams']]
    plt.plot(interest['Date'], interest['Streams'])
    plt.xlabel('Date')
    plt.ylabel('Streams')
    plt.xticks(rotation=60)
    plt.title(track + ' by ' + artist)
    plt.tight_layout()
    plt.style.use('fivethirtyeight')
    plt.margins(0.05)
    plt.show()

def get_data_artist(artist):
    '''This is to plot the number of streams vs date for any particular artist.'''
    interest = spotify_data.loc[spotify_data['Artist']==artist][['Date','Position','Streams']]
    interest1 = interest.groupby('Date', as_index=False)['Streams'].sum()
    plt.plot(interest1['Date'], interest1['Streams'])
    plt.xlabel('Date')
    plt.ylabel('Streams')
    plt.xticks(rotation=60)
    plt.title(artist)
    plt.tight_layout()
    plt.style.use('fivethirtyeight')
    plt.margins(0.05)
    plt.show()

get_data_track('You Need To Calm Down','Taylor Swift')
get_data_artist('Drake')

#Save spotify_data to local file
spotify_data.to_csv(r'C:\Users\user\Desktop\regional-global-daily-2years.csv')

#
#Question for analysis: 
#On Spotify, 50 songs with the highest streams will be featured in Global Top 50 Chart Playlist. Does the appearance of the song in the chart help boost its performance?
#I am going to create 3 different groups of data:
# 1. Group 1: Position in day 1 is between 51 and 55, day 2 between 46 and 50 (Global Top 50 exposure on day 2, not day 1)
# 2. Group 2: Position in day 1 is between 56 and 60, day 2 between 51 and 55 (no Global Top 50 exposure on day 1 & 2)
# 3. Group 3: Position in day 1 is between 46 and 50, day 2 between 45 and 41 (Global Top 50 exposure on day 1 & 2)
#Then, I am computing the change in position from day 2 to day 3, relative to the change from day 1 to day 2. I'll compare this among the 3 groups to see the Top 50 chart impact.
#
#Use 'spot' DataFrame to perform analysis, retain only the necessary columns for analysis. Add suffix '2' to the current date, position and streams.
#We are going to add the position and streams on the day before (suffix '1') and after (suffix '3'), so that we can compare the song performance day-to-day.
spot = pd.read_csv(r'C:\Users\user\Desktop\regional-global-daily-2years.csv')
spot['Date'] = spot['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) #change Date column to datetime format
spot = spot[['Track Name', 'Artist', 'Date', 'Position', 'Streams']]
spot.columns = ['Track','Artist','Date2','Position2','Stream2']

def get_values(x):
    '''This is to get the position and stream for a track by a particular artist on a certain date.'''
    track = x[0]
    artist = x[1]
    date = x[2]
    if not(spot.loc[(spot['Track']==track) & (spot['Artist']==artist) & (spot['Date2']==date)]['Position2'].empty):
        return pd.Series(spot.loc[(spot['Track']==track) & (spot['Artist']==artist) & (spot['Date2']==date)][['Position2','Stream2']].values[0])
    else: return pd.Series([np.nan, np.nan])

#Add the position and stream for that song on the day before
spot['Date1'] = spot['Date2'].apply(lambda x: x - relativedelta(days=1))
spot[['Position1','Stream1']] = spot[['Track','Artist','Date1']].apply(get_values, axis=1)

#Add the position and stream for that song on the day after
spot['Date3'] = spot['Date2'].apply(lambda x: x + relativedelta(days=1))
spot[['Position3','Stream3']] = spot[['Track','Artist','Date3']].apply(get_values, axis=1)

#Reorder the columns
spot = spot[['Track', 'Artist', 'Date1', 'Position1', 'Stream1', 'Date2', 'Position2', 'Stream2', 'Date3', 'Position3', 'Stream3']]

#Save spot to local file
spotify_data.to_csv(r'C:\Users\user\Desktop\spotify_data_for_analysis.csv')

#Calculate position/streams change from day 2 to day 3, relative to the change from day 1 to day 2
spot['rel_pos_change'] = -((spot['Position3'] - spot['Position2']) - (spot['Position2'] - spot['Position1']))
spot['rel_stream_change'] = (spot['Stream3'] - spot['Stream2']) - (spot['Stream2'] - spot['Stream1'])

#Create 3 different groups of data:
# 1. Group 1 (spot50): Position in day 1 is between 51 and 55, day 2 between 46 and 50 (Global Top 50 exposure on day 2, not day 1)
# 2. Group 2 (spot55): Position in day 1 is between 56 and 60, day 2 between 51 and 55 (no Global Top 50 exposure on day 1 & 2)
# 3. Group 3 (spot45): Position in day 1 is between 46 and 50, day 2 between 45 and 41 (Global Top 50 exposure on day 1 & 2)
spot50 = spot.loc[(spot['Position1'] <= 55) & (spot['Position1'] >= 51) & (spot['Position2'] <= 50) & (spot['Position2'] >= 46)]
spot50.dropna(subset=['rel_pos_change'], inplace=True)

spot55 = spot.loc[(spot['Position1'] <= 60) & (spot['Position1'] >= 56) & (spot['Position2'] <= 55) & (spot['Position2'] >= 51)]
spot55.dropna(subset=['rel_pos_change'], inplace=True)

spot45 = spot.loc[(spot['Position1'] <= 50) & (spot['Position1'] >= 46) & (spot['Position2'] <= 45) & (spot['Position2'] >= 41)]
spot45.dropna(subset=['rel_pos_change'], inplace=True)

#To get the feel of how the data looks like
#From here, we can see that spot50['rel_pos_change'] mean is still higher than that of spot55 and spot45. Does Global Top 50 Chart exposure help the track performance?
print(spot55.describe(), spot50.describe(), spot45.describe())

#Perform Exploratory Data Analysis (from the plots, a few outliers are spotted)
_ = plt.subplot(4,2,1)
_ = plt.hist(spot50['rel_pos_change'], alpha=0.5, bins=50, color='blue', normed=True)
_ = plt.hist(spot55['rel_pos_change'], alpha=0.5, bins=50, color='red', normed=True)
#_ = plt.xlabel('Relative Position Change', fontsize=8)
_ = plt.ylabel('Count', fontsize=8)
_ = plt.xticks(fontsize=5)
_ = plt.yticks(fontsize=5)
_ = plt.title('Position (51-55 To 46-50, Blue) VS (56-60 To 51-55, Red) Before Outlier Removal', fontsize=8)
_ = plt.margins(0.1)

_ = plt.subplot(4,2,2)
_ = plt.hist(spot50['rel_stream_change'], alpha=0.5, bins=50, color='blue', normed=True)
_ = plt.hist(spot55['rel_stream_change'], alpha=0.5, bins=50, color='red', normed=True)
#_ = plt.xlabel('Relative Stream Change', fontsize=8)
_ = plt.ylabel('Count', fontsize=8)
_ = plt.xticks(fontsize=5)
_ = plt.yticks(fontsize=5)
_ = plt.title('Stream (51-55 To 46-50, Blue) VS (56-60 To 51-55, Red) Before Outlier Removal', fontsize=8)
_ = plt.margins(0.1)

_ = plt.subplot(4,2,3)
_ = plt.hist(spot50['rel_pos_change'], alpha=0.5, bins=50, color='blue', normed=True)
_ = plt.hist(spot45['rel_pos_change'], alpha=0.5, bins=50, color='green', normed=True)
#_ = plt.xlabel('Relative Position Change', fontsize=8)
_ = plt.ylabel('Count', fontsize=8)
_ = plt.xticks(fontsize=5)
_ = plt.yticks(fontsize=5)
_ = plt.title('Position (51-55 To 46-50, Blue) VS (46-50 To 41-45, Green) Before Outlier Removal', fontsize=8)
_ = plt.margins(0.1)

_ = plt.subplot(4,2,4)
_ = plt.hist(spot50['rel_stream_change'], alpha=0.5, bins=50, color='blue', normed=True)
_ = plt.hist(spot45['rel_stream_change'], alpha=0.5, bins=50, color='green', normed=True)
#_ = plt.xlabel('Relative Stream Change', fontsize=8)
_ = plt.ylabel('Count', fontsize=8)
_ = plt.xticks(fontsize=5)
_ = plt.yticks(fontsize=5)
_ = plt.title('Stream (51-55 To 46-50, Blue) VS (46-50 To 41-45, Green) Before Outlier Removal', fontsize=8)
_ = plt.margins(0.1)

#Remove outlier from 3 different groups based on 1.5 interquartile range from 1st and 3rd quartile.
spot50_IQR = np.percentile(spot50['rel_pos_change'], 75) - np.percentile(spot50['rel_pos_change'], 25)
spot50 = spot50[(spot50['rel_pos_change'] >= (np.percentile(spot50['rel_pos_change'], 25) - (1.5 * spot50_IQR))) & \
(spot50['rel_pos_change'] <= (np.percentile(spot50['rel_pos_change'], 75) + (1.5 * spot50_IQR)))]

spot55_IQR = np.percentile(spot55['rel_pos_change'], 75) - np.percentile(spot55['rel_pos_change'], 25)
spot55 = spot55[(spot55['rel_pos_change'] >= (np.percentile(spot55['rel_pos_change'], 25) - (1.5 * spot55_IQR))) & \
(spot55['rel_pos_change'] <= (np.percentile(spot55['rel_pos_change'], 75) + (1.5 * spot55_IQR)))]

spot45_IQR = np.percentile(spot45['rel_pos_change'], 75) - np.percentile(spot45['rel_pos_change'], 25)
spot45 = spot45[(spot45['rel_pos_change'] >= (np.percentile(spot45['rel_pos_change'], 25) - (1.5 * spot45_IQR))) & \
(spot45['rel_pos_change'] <= (np.percentile(spot45['rel_pos_change'], 75) + (1.5 * spot45_IQR)))]

#Plot the data again after outlier removal.
_ = plt.subplot(4,2,5)
_ = plt.hist(spot50['rel_pos_change'], alpha=0.5, bins=50, color='blue', normed=True)
_ = plt.hist(spot55['rel_pos_change'], alpha=0.5, bins=50, color='red', normed=True)
#_ = plt.xlabel('Relative Position Change', fontsize=8)
_ = plt.ylabel('Count', fontsize=8)
_ = plt.xticks(fontsize=5)
_ = plt.yticks(fontsize=5)
_ = plt.title('Position (51-55 To 46-50, Blue) VS (56-60 To 51-55, Red) After Outlier Removal', fontsize=8)
_ = plt.margins(0.1)

_ = plt.subplot(4,2,6)
_ = plt.hist(spot50['rel_stream_change'], alpha=0.5, bins=50, color='blue', normed=True)
_ = plt.hist(spot55['rel_stream_change'], alpha=0.5, bins=50, color='red', normed=True)
#_ = plt.xlabel('Relative Stream Change', fontsize=8)
_ = plt.ylabel('Count', fontsize=8)
_ = plt.xticks(fontsize=5)
_ = plt.yticks(fontsize=5)
_ = plt.title('Stream (51-55 To 46-50, Blue) VS (56-60 To 51-55, Red) After Outlier Removal', fontsize=8)
_ = plt.margins(0.1)

_ = plt.subplot(4,2,7)
_ = plt.hist(spot50['rel_pos_change'], alpha=0.5, bins=50, color='blue', normed=True)
_ = plt.hist(spot45['rel_pos_change'], alpha=0.5, bins=50, color='green', normed=True)
_ = plt.xlabel('Relative Position Change', fontsize=8)
_ = plt.ylabel('Count', fontsize=8)
_ = plt.xticks(fontsize=5)
_ = plt.yticks(fontsize=5)
_ = plt.title('Position (51-55 To 46-50, Blue) VS (46-50 To 41-45, Green) After Outlier Removal', fontsize=8)
_ = plt.margins(0.1)

_ = plt.subplot(4,2,8)
_ = plt.hist(spot50['rel_stream_change'], alpha=0.5, bins=50, color='blue', normed=True)
_ = plt.hist(spot45['rel_stream_change'], alpha=0.5, bins=50, color='green', normed=True)
_ = plt.xlabel('Relative Stream Change', fontsize=8)
_ = plt.ylabel('Count', fontsize=8)
_ = plt.xticks(fontsize=5)
_ = plt.yticks(fontsize=5)
_ = plt.title('Stream (51-55 To 46-50, Blue) VS (46-50 To 41-45, Green) After Outlier Removal', fontsize=8)
_ = plt.margins(0.1)

_ = plt.subplots_adjust(hspace = 0.3)

figure = plt.gcf() # get current figure
figure.set_size_inches(12, 10)
plt.savefig(r'C:\Users\user\Desktop\data_analysis.png')
plt.show()

#To get the feel of how the data looks like after outlier removal
#From here, we can see that spot50['rel_pos_change'] mean is still higher than that of spot55 and spot45, which means that Global Top 50 Chart exposure may help the track performance.
#spot50 vs spot55 difference = 1.55 position (spot50 is better)
#spot50 vs spot45 difference = 0.86 position (spot50 is better)
print(spot55.describe(), spot50.describe(), spot45.describe())

#Perform bootstrap resampling analysis to measure how confident we are that the above observation is accurate.
#Null hypothesis_1: the mean of spot50 and spot55 relative position change are the same.
#Null hypothesis_2: the mean of spot50 and spot45 relative position change are the same.

def draw_bs_reps(data, func, size=1):
    ''''Draw bootstrap replicates.'''   
    #Initialize empty array
    bs_replicates = np.empty(size)
    #Generate replicates
    for i in range(size):
        bs_replicates[i] = func(np.random.choice(data, len(data)))
    return bs_replicates

#Comparing spot55 and spot50 data
spot50chg = spot50['rel_pos_change']
spot55chg = spot55['rel_pos_change']
diff_means_1 = np.mean(spot50chg) - np.mean(spot55chg) #1.55

#Compute mean of pooled data
mean_pool_1 = np.mean(np.concatenate([spot50chg, spot55chg])) #-3.51

#Generate shifted datasets for both with mean = mean_pool_1
spot50chgshift_1 = spot50['rel_pos_change'] - np.mean(spot50chg) + mean_pool_1
spot55chgshift = spot55['rel_pos_change'] - np.mean(spot55chg) + mean_pool_1

#Generate bootstrap replicates
bs_reps_spot50_1 = draw_bs_reps(spot50chgshift_1, np.mean, size=100000)
bs_reps_spot55 = draw_bs_reps(spot55chgshift, np.mean, size=100000)
bs_reps_1 = bs_reps_spot50_1 - bs_reps_spot55

#Compute p-values
p1 = np.sum(bs_reps_1 >= diff_means_1)/len(bs_reps_1)

#Comparing spot45 and spot50 data
spot50chg = spot50['rel_pos_change']
spot45chg = spot45['rel_pos_change']
diff_means_2 = np.mean(spot50chg) - np.mean(spot45chg) #0.86

#Compute mean of pooled data
mean_pool_2 = np.mean(np.concatenate([spot50chg, spot45chg])) #-3.11

#Generate shifted datasets for both with mean = mean_pool_2
spot50chgshift_2 = spot50['rel_pos_change'] - np.mean(spot50chg) + mean_pool_2
spot45chgshift = spot45['rel_pos_change'] - np.mean(spot45chg) + mean_pool_2

#Generate bootstrap replicates
bs_reps_spot50_2 = draw_bs_reps(spot50chgshift_2, np.mean, size=100000)
bs_reps_spot45 = draw_bs_reps(spot45chgshift, np.mean, size=100000)
bs_reps_2 = bs_reps_spot50_2 - bs_reps_spot45

#Compute p-values
p2 = np.sum(bs_reps_2 >= diff_means_2)/len(bs_reps_2)

print(p1, p2)
#p1 = 0 and p2 = 5e^-5 (both are close to 0, hence the null hypothesis can be rejected with 99.50% confidence interval)
#So, Spotify Global Top 50 Chart exposure can boost the track performance. By having the exposure, the average of position change is about 0.86 to 1.55 (in ranking) more compared to position change without the exposure.
