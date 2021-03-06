from board import D4
import adafruit_dht
import pandas as pd
import datetime
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--freq', default=1, help='frequency')
parser.add_argument('--period', default=6, help='period of sampling')
parser.add_argument('--output_path', default='WORK_DIR/Ex_Frank/LABS/recordings.csv')
args = parser.parse_args()


# Exercise 1: Read from a DHT sensor and generate csv
dht_device = adafruit_dht.DHT11(D4)
temperature = dht_device.temperature
humidity = dht_device.humidity
freq = args.freq
period = args.period
temps_list = []
hum_list = []
dates = []
times = []

for i in range(int(period/freq)):
    now = datetime.datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    hour = now.strftime("%H:%M:%S")
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    date = date_time.split(',')[0]
    hour = date_time.split(',')[1]
    dates.append(date)
    times.append(hour)
    temps_list.append(temperature)
    hum_list.append(humidity)
    time.sleep(5)

dates = pd.Series(dates)
hours = pd.Series(times)
temps = pd.Series(temps_list)
hums = pd.Series(hum_list)

pd.DataFrame({'Date': dates, 'Hour': hours, 'Temperature': temps, 'Humidity': hums})\
    .to_csv(args.output_path, header=False, index=False, index_label=None)

