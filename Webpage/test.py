import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import numpy as np
import json

# Read the country mapping from the JSON file
with open("./country_mapping.json", "r") as json_file:
    country_mapping = json.load(json_file)


# Read the CSV file into a DataFrame
login_dataframe = pd.read_csv("Dataset/login.csv")

# Remap the 'country' column using the country_mapping dictionary
login_dataframe['country'] = login_dataframe['country'].map(country_mapping).fillna(login_dataframe['country'])
login_dataframe['reg_date'] = pd.to_datetime(login_dataframe['reg_date'], unit='s')



# Read the daily reports CSV file into a DataFrame
daily_reports_df = pd.read_csv("Dataset/daily_report.csv")
daily_reports_df['record_time'] = pd.to_datetime(daily_reports_df['record_time'])



# Load and prepare the trades data frame
trades_dataframe = pd.read_csv("Dataset/trades.csv")
trades_dataframe['open_time'] = pd.to_datetime(trades_dataframe['open_time'], unit='s')
trades_dataframe['close_time'] = pd.to_datetime(trades_dataframe['close_time'], unit='s')



# Range of trades_dataframe
min_trade_date1 = trades_dataframe['open_time'].min()
max_trade_date1 = trades_dataframe['close_time'].max()



# Identify the minimum and maximum trade dates
min_trade_date = trades_dataframe['open_time'].min()
max_trade_date = trades_dataframe['close_time'].max()

# Filter the login dataframe to include only accounts registered within the trade dates range
filtered_login_dataframe = login_dataframe[(login_dataframe['reg_date'] >= min_trade_date) & (login_dataframe['reg_date'] <= max_trade_date)]

# Merge the filtered login data with the trades data
merged_df = pd.merge(trades_dataframe, filtered_login_dataframe, on='login', how='inner')





# Combine Daily Charts
daily_charts = glob.glob("Dataset/daily_chart/*.csv")
all_charts = pd.DataFrame()

for chart in daily_charts:
    name = Path(chart).stem
    csv = pd.read_csv(chart, index_col='date', parse_dates=True)
    all_charts[name] = csv['close']



# Convert all rates to be from AccountCurrency to USD and rename columns
usd_columns = [col for col in all_charts.columns if col.startswith('USD')]
all_charts_USD = all_charts.copy()
all_charts_USD[usd_columns] = 1 / all_charts_USD[usd_columns]
all_charts_USD.columns = [col.replace('USD', '') for col in all_charts_USD.columns]
all_charts_USD['USD'] = float(1)



# Converting Account Currency to USD
login_trades_rates = merged_df
login_trades_rates['trade_date'] = pd.to_datetime(pd.to_datetime(login_trades_rates['open_time']).dt.date)

all_charts_USD_stacked = all_charts_USD.stack()
login_trades_rates = pd.merge(login_trades_rates, all_charts_USD_stacked.rename('usd_rate'), how='left', left_on=['trade_date', 'account_currency'], right_index=True)

conversion_columns = ['commission', 'swaps', 'profit']
login_trades_USD = login_trades_rates

for column in conversion_columns:
    login_trades_USD[column] = login_trades_USD [column] * login_trades_USD['usd_rate']

merged_df = login_trades_USD






def infer_tp_sl_hit(row):
    if row['cmd'] == 0:  # Buy trade
        if row['close_price'] >= row['tp'] and row['tp'] > 0:
            return 'tp_hit'
        elif row['close_price'] <= row['sl'] and row['sl'] > 0:
            return 'sl_hit'
    elif row['cmd'] == 1:  # Sell trade
        if row['close_price'] <= row['tp'] and row['tp'] > 0:
            return 'tp_hit'
        elif row['close_price'] >= row['sl'] and row['sl'] > 0:
            return 'sl_hit'
    return 'none'





# Count the number of trades per account
total_trades_per_login = merged_df.groupby('login')['ticket'].count()

# Count the number and percentage of buy trades per account
buy_trades_per_login = merged_df[merged_df['cmd'] == 0].groupby('login')['ticket'].count()
percentage_buys = (buy_trades_per_login / total_trades_per_login * 100).fillna(0)

# Calculate various mean averages
average_volume_per_login = merged_df.groupby('login')['volume'].mean() 
average_volume_usd_per_login = merged_df.groupby('login')['volume_usd'].mean()
average_commission_per_login = merged_df.groupby('login')['commission'].mean()
average_swaps_per_login = merged_df.groupby('login')['swaps'].mean()
average_profit_per_login = merged_df.groupby('login')['profit'].mean()

# Calculate ratio of profitable trades
profitable_trades = merged_df[merged_df['profit'] > 0]
ratio_profitable_trades = profitable_trades.groupby('login').size() / total_trades_per_login.replace(0, pd.NA)
# ratio_profitable_trades = ratio_profitable_trades.fillna(0)

# Calculate profit and loss variability per account
profit_loss_variability = merged_df.groupby('login')['profit'].std()
# profit_loss_variability = profit_loss_variability.fillna(0)

# Calculate average trade duration per account
merged_df['trade_duration'] = (merged_df['close_time'] - merged_df['open_time']).dt.total_seconds()
average_trade_duration = merged_df.groupby('login')['trade_duration'].mean()

# Calculate average DPM per account
merged_df['DPM'] = merged_df['profit'] / (merged_df['volume_usd'] / 1e6)  # Converting volume from USD to million USD
average_dpm_per_login = merged_df.groupby('login')['DPM'].mean()

# Find the most common reason per account
reason_per_login = merged_df.groupby('login')['reason'].apply(lambda x: x.value_counts().idxmax())

# Trading Product Diversity: Total unique symbols traded per account
unique_symbols_traded = merged_df.groupby('login')['symbol'].nunique()

# Peak Trading Times: Most frequent trading hour per account
merged_df['trade_hour'] = merged_df['open_time'].dt.hour
peak_trading_times = merged_df.groupby('login')['trade_hour'].agg(lambda x: x.value_counts().idxmax())


# TP/SL calculations
# Add column to dataset whether tp or sl has been hit
merged_df['tp_sl_hit'] = merged_df.apply(infer_tp_sl_hit, axis=1)

# Calculate the TP and SL hit frequencies for each account
tp_hits = merged_df[merged_df['tp_sl_hit'] == 'tp_hit'].groupby('login').size()
sl_hits = merged_df[merged_df['tp_sl_hit'] == 'sl_hit'].groupby('login').size()

# Calculate the TP/SL hit frequency ratio (ensure no division by zero)
# Replace 0 with a small number to avoid division by zero or use np.where to handle 0 cases
tp_sl_hit_frequency_ratio = (tp_hits / sl_hits.replace(0, 1)).fillna(0)

# Calculate the average profit for trades where TP was hit
average_profit_tp = merged_df[merged_df['tp_sl_hit'] == 'tp_hit'].groupby('login')['profit'].mean()

# Calculate the average loss for trades where SL was hit
average_loss_sl = merged_df[merged_df['tp_sl_hit'] == 'sl_hit'].groupby('login')['profit'].mean()

# Calculate the Reward-to-Risk Ratio
# Note: Ensure no division by zero
reward_to_risk_ratio = (average_profit_tp / -average_loss_sl.replace(0, pd.NA)).fillna(0)


# Compile these metrics into a single dataframe
result_dataframe = pd.DataFrame({
    'Total_Trades': total_trades_per_login,
    'Buy_Percentage': percentage_buys,
    'Average_Volume': average_volume_per_login,
    'Average_Volume_USD': average_volume_usd_per_login,
    'Average_DPM': average_dpm_per_login,
    'Unique_Symbols_Traded': unique_symbols_traded,
    'Peak_Trading_Times': peak_trading_times,
    'Ratio_Profitable_Trades': ratio_profitable_trades,
    'Profit_Loss_Variability': profit_loss_variability,
    'Average_Trade_Duration': average_trade_duration,
    'TP/SL Hit Ratio': tp_sl_hit_frequency_ratio,
    'Reward_Risk_Ratio': reward_to_risk_ratio,
    'Most_Common_Trading_Method': reason_per_login,
    'Average_Commission': average_commission_per_login,
    'Average_Swaps': average_swaps_per_login,
    'Average_Profit': average_profit_per_login
})




final_dataset = pd.merge(login_dataframe, result_dataframe, on='login', how='inner')




# Calculate the first open time and last close time for each account
first_open_time_per_login = merged_df.groupby('login')['open_time'].min()
last_close_time_per_login = merged_df.groupby('login')['close_time'].max()

# Calculate longevity as the difference in days between the last close time and first open time
longevity_per_login = (last_close_time_per_login - first_open_time_per_login).dt.days

# Define the most recent month based on the latest trade in the dataset
most_recent_month_start = merged_df['close_time'].max().replace(day=1)

# Determine active accounts (trading in the most recent month)
active_accounts = merged_df[merged_df['close_time'] >= most_recent_month_start].groupby('login').size().index

# Mark accounts as active or inactive
longevity_per_login = longevity_per_login.to_frame(name='longevity')
longevity_per_login['active'] = longevity_per_login.index.isin(active_accounts)

# Exclude accounts that were registered in the most recent month
valid_accounts = login_dataframe[login_dataframe['reg_date'] < most_recent_month_start]

# Bin the longevity values
bins = [-1, 30, 90, 180, 270, 360, float('inf')]
longevity_per_login['longevity_bin'] = pd.cut(longevity_per_login['longevity'], bins=bins, labels=False)

# Exclude active accounts from all bins except for 360+
longevity_per_login = longevity_per_login[(longevity_per_login['active'] == False) | (longevity_per_login['longevity_bin'] == 5)]



# Merge longevity information with result_dataframe
result_dataframe = final_dataset.reset_index().merge(longevity_per_login, on='login', how='inner')

# Calculate trading frequency based on total number of trades and longevity
result_dataframe['Trading_Frequency'] = result_dataframe['Total_Trades'] / result_dataframe['longevity'].replace(0, 1)


# Filter rows where 'longevity_bin' is 5
filtered_df = result_dataframe[result_dataframe['longevity_bin'] == 5]

# Count the occurrences of True and False in the 'active' column for the filtered rows
active_counts = filtered_df['active'].value_counts()


# Group by 'login' and the month of 'record_time' to prepare for aggregation
daily_reports_df['month_year'] = daily_reports_df['record_time'].dt.to_period('M')


# Group by login and month_year and calculate average net_deposit and credit
monthly_averages = daily_reports_df.groupby(['login', 'month_year']).agg(
    average_net_deposit=('net_deposit', 'mean'),
    average_credit=('credit', 'mean')
).reset_index()

# Aggregate these monthly averages across all months to get an overall monthly average for each login
overall_monthly_averages = monthly_averages.groupby('login').agg(
    average_net_deposit=('average_net_deposit', 'mean'),
    average_credit=('average_credit', 'mean')
)


result_dataframe = result_dataframe.merge(overall_monthly_averages, on='login', how='left')



# Filter out the dataset
result_dataframe['Most_Common_Trading_Method'] = result_dataframe['Most_Common_Trading_Method'].apply(lambda x: x if x in [0, 1, 5] else 7)

# Rename the codes to strings
result_dataframe['Trading_Method'] = result_dataframe['Most_Common_Trading_Method'].map({0: 'Client', 1: 'Expert', 5: 'Mobile', 7:'Other'})
result_dataframe.drop('Most_Common_Trading_Method', axis=1, inplace=True)


# Remove reg_date
# Remove Total_Trades
# Move trading frequency to front, and active to back
result_dataframe.drop(['index', 'reg_date'], axis=1, inplace=True)


result_dataframe.fillna(0, inplace=True)


result_dataframe['country'] = result_dataframe['country'].astype('category')
result_dataframe['account_currency'] = result_dataframe['account_currency'].astype('category')
result_dataframe['Trading_Method'] = result_dataframe['Trading_Method'].astype('category')
result_dataframe['Peak_Trading_Times'] = result_dataframe['Peak_Trading_Times'].astype('category')


cols = list(result_dataframe.columns)

cols.remove('Trading_Frequency')
cols.remove('active')

# Insert 'Trading Frequency' at the new position
cols.insert(3, 'Trading_Frequency')

# Insert 'Active' at the new position
cols.insert(23, 'active')

cols = [col for col in cols if col not in ['longevity', 'longevity_bin']]
cols += ['longevity', 'longevity_bin']

# Reorder the DataFrame based on the new column list
modelling_df = result_dataframe[cols]

# Convert 'country', 'account_currency', and 'Trading_Method' to categorical
modelling_df['country'] = modelling_df['country'].astype('category')
modelling_df['account_currency'] = modelling_df['account_currency'].astype('category')
modelling_df['Trading_Method'] = modelling_df['Trading_Method'].astype('category')

# Convert 'active' column to categorical
modelling_df['active'] = modelling_df['active'].astype('category')

# Drop Total_Trades
modelling_df.drop(['Total_Trades'], axis=1, inplace=True)

# Remove specified columns and set 'longevity' as the target variable
features_df = modelling_df.drop(columns=['login', 'active', 'Unique_Symbols_Traded', 'Average_Volume', 'longevity', 'longevity_bin'])
y = modelling_df['longevity']