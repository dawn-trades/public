import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from lightweight_charts import Chart
import pytz
import json
import requests
from datetime import datetime, timedelta, time
import time as t
import pytz
from polygon import RESTClient
import threading
import pyperclip
import os
from dotenv import load_dotenv
load_dotenv()
from my_package import file_path

API_KEY = os.getenv('API_KEY')
client = RESTClient(API_KEY)

print('start')

# add on bar update to color new bars - for bar-replay feature

# should be possible to add/delete a line at entry, entry + x% stop level, open, close etc.

# ticker changes don't work on the data, it only loads from ticker change, stitch previous ticker data together
  
    # check if for the CIK the ticker has changed in the requested timespan
        # has to be done on processing level for the charting
            # check last ticker change date
            # check last ticker name
    
        # if ticker has changed get old ticker and stitch together

# maybe include ticker change, R/S, IPO, News info in a table in the chart?

##########################################################################################################################################################################

def get_trades(symbol, start_date, end_date, API_KEY, start_time_hours = 8, start_time_minutes = 0, end_time_hours = 8, end_time_minutes = 45):
    # Timezone for NYSE
    nyse_tz = pytz.timezone('America/New_York')

    # Convert start and end dates to datetime objects with NYSE timezone
    start_datetime = nyse_tz.localize(datetime.strptime(start_date.strftime("%Y-%m-%d"), "%Y-%m-%d") + timedelta(hours=start_time_hours, minutes=start_time_minutes))
    end_datetime = nyse_tz.localize(datetime.strptime(end_date.strftime("%Y-%m-%d"), "%Y-%m-%d") + timedelta(hours=end_time_hours, minutes=end_time_minutes))

    # Convert to timestamps in nanoseconds
    start_timestamp = int(start_datetime.timestamp() * 1e9)
    end_timestamp = int(end_datetime.timestamp() * 1e9)
    
    base_url = f"https://api.polygon.io/v3/trades/{symbol}"
    all_trades = []
    next_url = None

    # Initial URL with start date
    url = f"{base_url}?timestamp.gte={start_timestamp}&timestamp.lte={end_timestamp}&order=asc&limit=50000&apiKey={API_KEY}"

    while url:
        response = requests.get(url)

        data = response.json()
        
        trades = data['results']
        
        all_trades.extend(trades)
        
        next_url = data.get('next_url', None)
        
        if not next_url:
            break
        
        url = f"{next_url}&apiKey={API_KEY}"

    return all_trades

def filter_trades(trades, max_deviation=2e9):  # deviation in nanoseconds
    filtered_trades = [trade for trade in trades if abs(trade['sip_timestamp'] - trade['participant_timestamp']) <= max_deviation]
    return filtered_trades

def aggregate_trades_to_bars(trades, cutoff=100):
    df = pd.DataFrame(trades)

    # Replace with actual column names from the API response
    timestamp_column = 'sip_timestamp'  # Replace with actual timestamp column name
    price_column = 'price'  # Replace with actual price column name
    volume_column = 'size'  # Replace with actual volume column name

    if all(column in df.columns for column in [timestamp_column, price_column, volume_column]):
        # Converting nanosecond timestamps to datetime
        df['timestamp'] = pd.to_datetime(df[timestamp_column], unit='ns')

        # Localize the timestamp to UTC and then convert to NYSE timezone
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

        df.set_index('timestamp', inplace=True)
        
        df['price_volume'] = df[price_column] * df[volume_column]

        # Aggregating price, volume, and trade count
        bars = df.resample('min').agg({
            price_column: 'ohlc',  # OHLC for price
            volume_column: 'sum',  # Sum of volume
            timestamp_column: 'count',  # Count of trades
            'price_volume': 'sum'
        })

        # Renaming columns for clarity
        bars.columns = bars.columns.map('_'.join)
    else:
        raise KeyError(f"Required columns not found in the trades data.")

    bars = bars[bars['size_size'] >= cutoff]
    
    bars = bars.reset_index()
    
    bars['timestamp'] = bars['timestamp'].dt.tz_localize(None)
    
    bars = bars.rename(columns={'price_open': 'open', 'price_close': 'close', 'price_high': 'high', 'price_low': 'low', 'sip_timestamp_sip_timestamp':'n', 'size_size':'volume', 'price_volume_price_volume':'vw', 'timestamp':'time'})

    bars['vw'] = bars['vw'] / bars['volume']
    
    bars = bars.drop(columns={'n'})
    
    return bars

def get_setup_data_uni_trades(setup, multiplier, timespan, lookback = 20, build_all_candles = False, replay_mode = False):
    scan_row = scan.loc[[setup]].reset_index(drop=True)
    ticker = scan_row.loc[0]['ticker']
    start_date = scan_row.loc[0][f'T_{lookback}_date']
    end_date = scan_row.loc[0]['date']
    multiplier = multiplier
    timespan = timespan

    aggs = client.get_aggs(
        ticker = ticker,
        multiplier = multiplier,
        timespan = timespan,
        from_ = start_date,
        to = end_date,
        raw = True,
        limit=20000
        )
    
    df = (json.loads(aggs.data))["results"]
    df = pd.DataFrame(df)

    df = df.rename(columns={'v': 'volume', 't':'time', 'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low'})
    df = df[['time', 'open', 'high', 'low', 'close', 'volume', 'vw']]
    df['time'] = pd.to_datetime(df['time'],unit='ms').dt.tz_localize(pytz.timezone('UTC')).dt.tz_convert(pytz.timezone('US/Eastern'))
    df['time'] = df['time'].dt.tz_localize(None)

    trades_date = pd.to_datetime(end_date) + pd.Timedelta(hours=8, minutes=45)

    # Filter DataFrame for entries on 'trades_date'
    df_on_date = df[(pd.to_datetime(df['time']) >= (pd.to_datetime(end_date) + pd.Timedelta(hours=4, minutes=0)))]

    if timespan == "minute" and not build_all_candles and any(pd.to_datetime(df_on_date['time']) < trades_date):
        trades = get_trades(ticker, end_date, end_date, API_KEY)
        filtered_trades = filter_trades(trades)
        vwap_df = df.copy() # copy the old df to base VWAP calculation off
        if len(filtered_trades) > 0:
            minute_bars = aggregate_trades_to_bars(filtered_trades)
            
            df.set_index('time', inplace=True)
            minute_bars.set_index('time', inplace=True)

            # Update the 'minute_df' with the matching 'time' data from 'trades_minute_df'
            df.update(minute_bars)

            # Reset index to view the updated data
            df = df.reset_index()
    elif any(pd.to_datetime(df_on_date['time']) < trades_date):
        trades = get_trades(ticker, end_date, end_date, API_KEY, start_time_hours = 4)
        filtered_trades = filter_trades(trades)
        minute_bars = aggregate_trades_to_bars(filtered_trades, cutoff=1)
        
        # df.set_index('time', inplace=True)
        # minute_bars.set_index('time', inplace=True)

        # # Update the 'minute_df' with the matching 'time' data from 'trades_minute_df'
        # df.update(minute_bars)

        # # Reset index to view the updated data
        # df = df.reset_index()
        
        # Ensure 'time' is set as the index for both DataFrames
        vwap_df = df.copy() # copy the old df to base VWAP calculation off
        df.set_index('time', inplace=True)
        minute_bars.set_index('time', inplace=True)

        # Update the 'df' with the matching 'time' data from 'minute_bars'
        df.update(minute_bars)

        # Identify rows in 'minute_bars' that are not in 'df'
        additional_data = minute_bars.loc[~minute_bars.index.isin(df.index)]

        # Concatenate these additional rows to 'df'
        df = pd.concat([df, additional_data])

        # Sort 'df' by the index (which is 'time')
        df.sort_index(inplace=True)

        # Reset the index if necessary
        df = df.reset_index()
    else:
        vwap_df = df.copy()   
        
    if replay_mode: 
        time_object = scan_row.loc[0]['para']['60min']['04:00 - 06:59_move_start_time'].time()
        comparison_time = datetime.combine(end_date, time_object)
        df = df[df['time'] < comparison_time]
        vwap_df = vwap_df[vwap_df['time'] < comparison_time]
        trades = get_trades(ticker, end_date, end_date, API_KEY, start_time_hours=time_object.hour, start_time_minutes=time_object.minute, end_time_hours=9, end_time_minutes=30)
        filtered_trades = filter_trades(trades)
        filtered_trades_df = pd.DataFrame(filtered_trades)
        filtered_trades_df['time'] = pd.to_datetime(filtered_trades_df['sip_timestamp'], unit='ns')
        filtered_trades_df['time'] = filtered_trades_df['time'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
        filtered_trades_df['timedelta'] = filtered_trades_df['time'].shift(-1) - filtered_trades_df['time']
        filtered_trades_df.rename(columns={'size': 'volume'}, inplace=True)
        filtered_trades_df = filtered_trades_df[['time', 'price', 'volume', 'timedelta',]]
            
        return (
            df,  # 0
            scan_row, # 1
            multiplier,  # 2
            timespan,  # 3
            setup,  # 4
            filtered_trades_df,  # 5
            vwap_df, # 6
        )

    else:
        return (
            df,  # 0
            scan_row, # 1
            multiplier,  # 2
            timespan,  # 3
            setup,  # 4
            'filtered_trades_df',  # 5 placeholder replay mode
            vwap_df, # 6
        )

def calc_vwap(df):
    # Ensure 'time' column is in datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Store the original order of the DataFrame
    original_order = df.index.copy()
    
    # Check if there's more than one unique date in the 'time' column
    unique_dates = df['time'].dt.date.nunique()

    if unique_dates > 1:
    # If there are multiple dates, perform the groupby operation
        df['vwap'] = df.groupby(df['time'].dt.date).apply(
        lambda x: (x['vw'] * x['volume']).cumsum() / x['volume'].cumsum()
        ).reset_index(level=0, drop=True)
    else:
    # Handle the case where there is only one date (or none)
    # This could be just a direct calculation, or a different form of handling
        df['vwap'] = (df['vw'] * df['volume']).cumsum() / df['volume'].cumsum()

    # Restore the original order of the DataFrame
    df = df.loc[original_order]

    return df

def T_0_PM_background_4(df, date):
    filtered_df = df[(df['time'].dt.date == pd.to_datetime(date).date()) & 
                 (df['time'].dt.time >= pd.to_datetime('04:00').time()) & 
                 (df['time'].dt.time <= pd.to_datetime('06:59').time())].copy()
    if not filtered_df.empty:
        filtered_df.loc[:, 'PM'] = 0
    return filtered_df

def T_0_PM_background_7(df, date):
    filtered_df = df[(df['time'].dt.date == pd.to_datetime(date).date()) & 
                 (df['time'].dt.time >= pd.to_datetime('07:00').time()) & 
                 (df['time'].dt.time <= pd.to_datetime('09:29').time())].copy()
    if not filtered_df.empty:
        filtered_df.loc[:, 'PM'] = 0
    return filtered_df

def T_0_AH_background(df, date):
    filtered_df = df[(df['time'].dt.date == pd.to_datetime(date).date()) & 
                 (df['time'].dt.time >= pd.to_datetime('16:00').time()) & 
                 (df['time'].dt.time <= pd.to_datetime('19:59').time())].copy()
    if not filtered_df.empty:
        filtered_df.loc[:, 'AH'] = 0
    return filtered_df

def T_1_PM_background(df, date):
    filtered_df = df[(df['time'].dt.date == pd.to_datetime(date).date()) & 
                 (df['time'].dt.time >= pd.to_datetime('04:00').time()) & 
                 (df['time'].dt.time <= pd.to_datetime('09:29').time())].copy()
    if not filtered_df.empty:
        filtered_df.loc[:, 'PM'] = 0
    return filtered_df

def T_1_AH_background(df, date):
    filtered_df = df[(df['time'].dt.date == pd.to_datetime(date).date()) & 
                 (df['time'].dt.time >= pd.to_datetime('16:00').time()) & 
                 (df['time'].dt.time <= pd.to_datetime('19:59').time())].copy()
    if not filtered_df.empty:
        filtered_df.loc[:, 'AH'] = 0
    return filtered_df

def next_button(chart):
    complete_stop_replay()
    global buffer_setup_data_1, buffer_setup_1, setup
    setup += 1
    
    update_chart(buffer_setup_data_1)
    
    buffer_setup_1 += 1
    buffer_setup_data_1 = get_setup_data_uni_trades(buffer_setup_1, 1, 'minute')

def previous_button(chart):
    global buffer_setup_data_1, buffer_setup_1, setup
    
    complete_stop_replay()
    
    if setup >= 1: setup -= 1
    
    setup_data = get_setup_data_uni_trades(setup, 1, 'minute')
    update_chart(setup_data)
    
    buffer_setup_1 = setup + 1
    buffer_setup_data_1 = get_setup_data_uni_trades(buffer_setup_1, 1, 'minute')

def on_timeframe_selection(chart):
    complete_stop_replay()
    
    if chart.topbar['timeframe'].value == '1day':
        setup_data = get_setup_data_uni_trades(setup, 1, 'day', lookback = 180)
        update_chart(setup_data, timeframe='1day')
    if chart.topbar['timeframe'].value == '1min':
        setup_data = get_setup_data_uni_trades(setup, 1, 'minute')
        update_chart(setup_data)
    if chart.topbar['timeframe'].value == '15min':
        setup_data = get_setup_data_uni_trades(setup, 15, 'minute', lookback = 60)
        update_chart(setup_data, timeframe='15min')
    if setup_data[0].empty:
        print('no Data')

def build_all_candles(chart):
    if not stop_event.is_set():
        stop_event.set()
    setup_data = get_setup_data_uni_trades(setup, 1, 'minute', 1, build_all_candles=True)
    update_chart(setup_data)

def start_replay_mode(chart):
    stop_event.clear()
    global complete_stop_requested, speedup_factor
    complete_stop_requested = False
    speedup_factor = 1
    setup_data = get_setup_data_uni_trades(setup, 1, 'minute', replay_mode=True)
    
    update_chart(setup_data, replay_mode=True)

def run_replay(df, pcl, PM_o, date):
    global stop_event, resume_event, complete_stop_requested, speedup_factor
    move_volume = 0
    for i, tick in df.iterrows():
        while stop_event.is_set():
            # print("Replay paused. Waiting to resume.")
            resume_event.wait(timeout=0.5)  # Wait for a short period before checking again
            if complete_stop_requested:
                # print("Replay completely stopped during pause.")
                return

        if complete_stop_requested:
            # print("Replay completely stopped.")
            return

        stop_event.clear()
        resume_event.clear()

        chart.update_from_tick(tick, cumulative_volume=True)
        move_volume += tick["volume"]
        
        chart.topbar['change_pcl'].set(f'PCL: {int((tick["price"] / pcl - 1) * 100)}%')
        chart.topbar['change_PM_o'].set(f'PM: {int((tick["price"] / PM_o - 1) * 100)}%')
        
        if move_volume >= 1e6:
            chart.topbar['volume'].set(f'Vol: {round(move_volume / 1e6, 2)}M')
        else:
            chart.topbar['volume'].set(f'Vol: {round(move_volume / 1e3, 2)}K')

        with speedup_lock:
            current_speedup = speedup_factor
        
        if tick['time'] > pd.Timestamp(f'{date} 09:29:00'):
            stop_event.set()
            
        sleep_time = min(tick['timedelta'].total_seconds(), 1) / current_speedup
        time.sleep(sleep_time)

def set_replay_speed_1x(chart):
    global speedup_factor

    with speedup_lock:
        speedup_factor = 1

def set_replay_speed_3x(chart):
    global speedup_factor

    with speedup_lock:
        speedup_factor = 3

def set_replay_speed_6x(chart):
    global speedup_factor

    with speedup_lock:
        speedup_factor = 6

def set_replay_speed_60x(chart):
    global speedup_factor

    with speedup_lock:
        speedup_factor = 60

def stop_resume_replay(chart):
    global stop_event, resume_event, complete_stop_requested

    if not complete_stop_requested:
        if stop_event.is_set():
            # If currently stopped, resume the replay
            resume_event.set()
            stop_event.clear()
            chart.topbar['stop_resume_replay'].set('stop')
            #print("Replay resumed.")
        else:
            # If currently running, pause the replay
            stop_event.set()
            chart.topbar['stop_resume_replay'].set('resume')
            #print("Replay paused. Waiting to resume.")

def complete_stop_replay():
    global complete_stop_requested
    complete_stop_requested = True
    stop_event.set()
    chart.topbar['stop_resume_replay'].set('stop')

def copy_setup_name(chart):
    pyperclip.copy(f'{chart.topbar["setup_name"].value}_{filter}')

def get_center_time(data):
    visible_range_left_bars = 200
    visible_range_right_bars = 300
    if filter == 'all':
        if data.loc[0]['PM_h'] > data.loc[0]['h']:
            center_time = pd.to_datetime(datetime.combine(data.loc[0]['date'], data.loc[0]['PM_h_dt']))
        else:
            center_time = pd.to_datetime(datetime.combine(data.loc[0]['date'], data.loc[0]['RTH_h_dt'])) 
    elif filter == 'epm' or filter == 'lpm' or filter == 'pm_pop':
        center_time = pd.to_datetime(datetime.combine(data.loc[0]['date'], data.loc[0]['PM_h_dt']))
    elif filter == 'ah':
        visible_range_left_bars = 250
        visible_range_right_bars = 150
        center_time = pd.to_datetime(datetime.combine(data.loc[0]['date'], data.loc[0]['PM_start']))
    elif filter == '7am':
        center_time = pd.to_datetime(datetime.combine(data.loc[0]['date'], time(7, 00)))
    elif filter == '4am_gap':
        center_time = pd.to_datetime(datetime.combine(data.loc[0]['date'], time(4, 00)))
    elif filter == 'pop':
        visible_range_left_bars = 60
        visible_range_right_bars = 120
        center_time = pd.to_datetime(datetime.combine(data.loc[0]['date'], data.loc[0]['PM_h_dt']))
    return center_time, visible_range_left_bars, visible_range_right_bars

def update_chart(setup_data, replay_mode = False, timeframe = "minute"):
    
    if timeframe == '1day':
        chart.watermark(text = f'{setup_data[1].loc[0]["ticker"]}, 1D', font_size = 80, color = '#2a2e39')
    elif timeframe == '15min':
        chart.watermark(text = f'{setup_data[1].loc[0]["ticker"]}, 15', font_size = 80, color = '#2a2e39')
    else:
        chart.watermark(text = f'{setup_data[1].loc[0]["ticker"]}, {setup_data[2]}', font_size = 80, color = '#2a2e39')
        
    chart.legend(visible=True, ohlc=True, percent=True, lines=False ,color='grey', font_size= 12, font_family='Arial',)
    
    chart.topbar['setup_name'].set(setup_data[1].loc[0]["key"])
    chart.topbar['sec_type'].set(setup_data[1].loc[0]["type"])
    
    if setup_data[1].loc[0]["market_cap"] < 1e7:
        chart.topbar['mcap'].set(f'{round(setup_data[1].loc[0]["market_cap"]/ 1e6, 2)}M')
    elif setup_data[1].loc[0]["market_cap"] < 1e8:
        chart.topbar['mcap'].set(f'{round(setup_data[1].loc[0]["market_cap"]/ 1e6, 1)}M')
    elif setup_data[1].loc[0]["market_cap"] < 1e9:
        chart.topbar['mcap'].set(f'{int(setup_data[1].loc[0]["market_cap"]/ 1e6)}M')
    else:
        chart.topbar['mcap'].set(f'{round(setup_data[1].loc[0]["market_cap"]/ 1e9, 2)}B')
    
    # chart.topbar['gap_stat'].set(f'gap: {setup_data[1].loc[0]["gap"]}%')
    # chart.topbar['PM_gap_stat'].set(f'PM gap: {setup_data[1].loc[0]["PM_gap"]}%')
    # chart.topbar['AH_dv_stat'].set(f'AH v: ${round(setup_data[1].loc[0]["AH_dv_T_1"]/1000000,2)}m')
    
    chart.topbar['timeframe'].set('1min')

    T__0_PM_background_4.set(T_0_PM_background_4(setup_data[0], setup_data[1].loc[0]["date"]))
    T__0_PM_background_7.set(T_0_PM_background_7(setup_data[0], setup_data[1].loc[0]["date"]))
    T__0_AH_background.set(T_0_AH_background(setup_data[0], setup_data[1].loc[0]["date"]))
    
    T__1_PM_background.set(T_1_PM_background(setup_data[0], setup_data[1].loc[0]["T_1_date"]))
    T__1_AH_background.set(T_1_AH_background(setup_data[0], setup_data[1].loc[0]["T_1_date"]))
    
    vwap.set(calc_vwap(setup_data[6]))
    
    chart.set(setup_data[0])
    chart.fit()

    if replay_mode:
        mid_time = get_center_time(setup_data[1])
        df_sorted = setup_data[0].sort_values('time')
        
        # Convert mid_time to a Timestamp if it's not already one
        mid_time = pd.to_datetime(mid_time)

        # Find the index for mid_time and adjust indices for start_time and end_time
        mid_index = min(df_sorted['time'].searchsorted(mid_time[0], side='left'), len(df_sorted) - 1)
        start_index, end_index = max(mid_index - mid_time[1], 0), min(mid_index + mid_time[2], len(df_sorted) - 1)

        # Set the visible range on the chart
        chart.set_visible_range(start_time=df_sorted.iloc[start_index]['time'], end_time=df_sorted.iloc[end_index]['time'])
        
        loop_thread = threading.Thread(target=run_replay, args=(setup_data[13], setup_data[1].loc[0]["pcl"], setup_data[1].loc[0]["PM_o"], setup_data[1].loc[0]["date"]))
        loop_thread.start()
        if complete_stop_requested:
            loop_thread.join()
    else:
        mid_time = get_center_time(setup_data[1])
        df_sorted = setup_data[0].sort_values('time')

        # Find the index for mid_time and adjust indices for start_time and end_time
        mid_index = min(df_sorted['time'].searchsorted(mid_time[0], side='left'), len(df_sorted) - 1)
        start_index, end_index = max(mid_index - mid_time[1], 0), min(mid_index + mid_time[2], len(df_sorted) - 1)

        # Set the visible range on the chart
        chart.set_visible_range(start_time=df_sorted.iloc[start_index]['time'], end_time=df_sorted.iloc[end_index]['time'])

# scan = pd.read_parquet(file_path.concat_para)
scan = pd.read_parquet('/Users/Daniel/Documents/coding/fin_data/processed_data/concat/concat_para_filter.parquet')

def filter_scan(option, ticker_symbol=""):
    if option == "epm":
        return scan[(scan['para'].apply(lambda x: x.get('60min', {}).get('04:00 - 06:59_up_move', 0) > 30))
                    & (scan['PM_dv'] > .5e6)]
    if option == "lpm":
        return scan[(scan['para'].apply(lambda x: x.get('60min', {}).get('07:00 - 09:29_up_move', 0) > 30))
                    & (scan['PM_dv'] > .5e6)]
    elif option == "all":
        return scan[((scan['para'].apply(lambda x: x.get('60min', {}).get('04:00 - 06:59_up_move', 0) >= 25))
                    & (scan['PM_dv'] >= .5e6))
                    | ((scan['para'].apply(lambda x: x.get('60min', {}).get('07:00 - 09:29_up_move', 0) >= 25))
                    & (scan['PM_dv'] >= .5e6))
                    | ((scan['para'].apply(lambda x: x.get('60min', {}).get('09:30 - 15:59_up_move', 0) >= 25))
                    & (scan['RTH_dv'] >= 2e6))]
    elif option == "pop":
        return scan[(scan['ticker'] == ticker_symbol) & (scan['d_volume'] >= 5e6)]
    elif option == "tliq": # liquidation or big fade the day before
        return scan[((scan['RTH_lah_T_1'] / scan['h_T_1'] - 1) * 100 <= -20)
                    & (scan['RTH_dv_T_1'] > 1e6)]
    elif option == "ah":
        return scan[(scan['AH_dv_T_1'] >= 1e6)
                    & ((scan['AH_h_T_1'] / scan['AH_l_T_1']) >= 1.5)
                    & (scan['RTH_dv_T_1'] >= 1e7)]
    elif option == "7am": 
        return scan[(scan['7am_up'] >= 20)
                    & (scan['para'].apply(lambda x: x.get('60min', {}).get('04:00 - 06:59_up_move', 0) > 30))
                    & (scan['PM_dv'] > .5e6)]
    elif option == "4am_gap": 
        return scan[(scan['PM_gap'] >= 25)
                    & (scan['AH_dv_T_1'] > .5e6)]
    elif option == "pm_pop": 
        return scan[((scan['para'].apply(lambda x: x.get('30min', {}).get('04:00 - 06:59_up_move', 0) >= 30))
                    & (scan['PM_dv'] >= .5e6))
                    | ((scan['para'].apply(lambda x: x.get('30min', {}).get('07:00 - 09:29_up_move', 0) >= 30))
                    & (scan['PM_dv'] >= .5e6))]
    else:
        return scan

ticker = 'POL' # if pop scan
filter = 'pm_pop' # epm, lpm, all, pop, tliq, ah, 7am, 4am_gap, pm_pop,
# to add: outlier RTH, swing squeeze (close weak, no AH squez), unhalt, single candle, strong into open, early pm squeeze, 

print(filter)

scan = filter_scan(option=filter, ticker_symbol=ticker)

scan = scan.drop_duplicates(subset='key')
scan = scan.sort_values(['date', 'd_volume'], ascending=False)

scan = scan.reset_index(drop=True)
setup = 0

# for handling the replay mode
stop_event = threading.Event()
resume_event = threading.Event()
complete_stop_requested = False

# for speeding up and slowing down replay playback
speedup_lock = threading.Lock()
speedup_factor = 1

#setup_data = get_setup_data(setup)
setup_data = get_setup_data_uni_trades(setup, 1, 'minute')

# store the data for the next setup to quickly load
buffer_setup_1 = setup + 1
buffer_setup_data_1 = get_setup_data_uni_trades(buffer_setup_1, 1, 'minute')
stop_replay_button = False

if __name__ == '__main__':
    try:
        chart = Chart(width=1450, height=1060, maximize=True, toolbox=False, title=f'Polygon Charts', scale_candles_only=True,)
        # chart.legend(visible=True, ohlc=True, lines=False, color='grey', font_size= 12, font_family='Arial',)
        
        vwap = chart.create_line(name='vwap', style='solid', width=1, color='rgba(210, 212, 63, 0.65)', price_line=False, price_label=False)
        
        T__0_PM_background_4 = chart.create_line(name='PM', style='solid', width=10000, color='rgba(240, 158, 50, 0.06)', price_line=False, price_label=False)
        T__0_PM_background_7 = chart.create_line(name='PM', style='solid', width=10000, color='rgba(255, 255, 255, 0.04)', price_line=False, price_label=False)
        T__0_AH_background = chart.create_line(name='AH', style='solid', width=10000, color='rgba(77, 118, 238, 0.06)', price_line=False, price_label=False)
        
        T__1_PM_background = chart.create_line(name='PM', style='solid', width=10000, color='rgba(240, 158, 50, 0.06)', price_line=False, price_label=False)
        T__1_AH_background = chart.create_line(name='AH', style='solid', width=10000, color='rgba(77, 118, 238, 0.06)', price_line=False, price_label=False)

        chart.price_line(label_visible=True, line_visible=False)
        chart.layout(background_color='#0f131b')
        chart.grid(vert_enabled=False, horz_enabled=False)
        chart.watermark(text=f'{setup_data[1].loc[0]["ticker"]}, {setup_data[2]}', font_size=80, color='#2a2e39')
        
        chart.volume_config(up_color='#438F80', down_color='#b44e4a',)
        chart.candle_style(up_color='#26a69a', down_color='#ef5350')
        chart.crosshair(vert_color='#5d606b', horz_color='#5d606b')
        chart.price_scale(border_color='#434651', border_visible=True, scale_margin_top=0.025)
        
        # functional buttons
        # chart.topbar.button('setup_count', f'{setup}', func=next_button)
        chart.topbar.button('setup_name', f'{setup_data[1].loc[0]["key"]}', func=copy_setup_name)
        chart.topbar.switcher('timeframe', ('1min', '15min', '1day'), default='1min', func=on_timeframe_selection)
        chart.topbar.button('build_all_candles', 'build all', func=build_all_candles)
        chart.topbar.button('previous_button', '◀ previous', func=previous_button)
        chart.topbar.button('next_button', 'next ▶', func=next_button)
        chart.topbar.button('replay_mode', 'start replay', func=start_replay_mode)
        chart.topbar.button('stop_resume_replay', 'stop', func=stop_resume_replay)
        chart.topbar.button('replay_speed_1x', '1x ▶', func=set_replay_speed_1x)
        chart.topbar.button('replay_speed_3x', '3x ▶▶', func=set_replay_speed_3x)
        chart.topbar.button('replay_speed_6x', '6x ▶▶▶', func=set_replay_speed_6x)
        chart.topbar.button('replay_speed_60x', '60x ▶▶▶▶', func=set_replay_speed_60x)
        
        # stat display buttons
        chart.topbar.button('sec_type', f'{setup_data[1].loc[0]["type"]}', )
        chart.topbar.button('mcap', f'{int(setup_data[1].loc[0]["market_cap"]/ 1e6)}M', )
        chart.topbar.button('change_pcl', f'change: {int(setup_data[1].loc[0]["pcl"])}%', )
        chart.topbar.button('change_PM_o', f'PM: {int(setup_data[1].loc[0]["PM_o"])}%', )
        chart.topbar.button('volume', f'vol: {round(setup_data[1].loc[0]["volume"] / 1e6, 2)}M', )
        # chart.topbar.button('gap_stat', f'gap: {setup_data[1].loc[0]["gap"]}',)
        # chart.topbar.button('PM_gap_stat', f'PM gap: {setup_data[1].loc[0]["PM_gap"]}',)
        # chart.topbar.button('AH_dv_stat', f'AH $ vol: {round(setup_data[1].loc[0]["AH_dv_T_1"]/1000000,2)}m',)
        
        vwap.set(calc_vwap(setup_data[6]))
        
        T__0_PM_background_4.set(T_0_PM_background_4(setup_data[0], setup_data[1].loc[0]["date"]))
        T__0_PM_background_7.set(T_0_PM_background_7(setup_data[0], setup_data[1].loc[0]["date"]))
        T__0_AH_background.set(T_0_AH_background(setup_data[0], setup_data[1].loc[0]["date"]))
        
        T__1_PM_background.set(T_1_PM_background(setup_data[0], setup_data[1].loc[0]["T_1_date"]))
        T__1_AH_background.set(T_1_AH_background(setup_data[0], setup_data[1].loc[0]["T_1_date"]))
        
        chart.legend(visible=True, ohlc=True, lines=False, color='grey', font_size= 12, font_family='Arial',)
        
        chart.fit()
        chart.set(setup_data[0])

        mid_time = get_center_time(setup_data[1])
        df_sorted = setup_data[0].sort_values('time')
        
        # Find the index for mid_time and adjust indices for start_time and end_time
        mid_index = min(df_sorted['time'].searchsorted(mid_time[0], side='left'), len(df_sorted) - 1)
        start_index, end_index = max(mid_index - mid_time[1], 0), min(mid_index + mid_time[2], len(df_sorted) - 1)

        # Set the visible range on the chart
        chart.set_visible_range(start_time=df_sorted.iloc[start_index]['time'], end_time=df_sorted.iloc[end_index]['time'])
        
        chart.show(block=True)

    finally:
        complete_stop_replay() # This code executes when exiting the GUI
        print(f'Shut down at setup: {setup}')
        t.sleep(1)
