# this program gathers sensor data

from DIPPID import SensorUDP
import time
import pandas as pd
import os
import sys

PORT = 5700
sensor = SensorUDP(PORT)

ACTIVITIES = ["running", "rowing", "lifting", "jumpingjacks"]
COLUMNS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
MAX_DATA_POINTS = 10000 #should be around 10 seconds
NAME = "emma"

measuring_started = False

#--- let user choose activity
for i in range(0, len(ACTIVITIES)):
    print("Activity id ", i, " - ", ACTIVITIES[i])

print('select activity:')
activity = ACTIVITIES[int(input())]

#--- data handling
#create folder named NAME, if it does not exist
if not os.path.exists(NAME):
    os.makedirs(NAME)

#used GPT for helping with this method
def safe_to_csv(df):
    #get current activity_count
    existing_activities = [filename for filename in os.listdir(NAME) if filename.startswith(f"{NAME}-{activity}-")]
    activity_count = len(existing_activities) + 1 if existing_activities else 1
    #generate file name and store
    filename = f"{NAME}-{activity}-{activity_count}.csv"
    df.to_csv(os.path.join(NAME, filename), index=False)

#--- start activity
print("Press button 1 to start " + activity + "!")

def get_data() -> list:
    if(sensor.has_capability('accelerometer')):
        accelerometer_data = sensor.get_value('accelerometer')
        acc_x = accelerometer_data['x']
        acc_y = accelerometer_data['y']
        acc_z = accelerometer_data['z']
    else: return
    if(sensor.has_capability('gyroscope')):
        gyroscope_data = sensor.get_value('gyroscope')
        gyro_x = gyroscope_data['x']
        gyro_y = gyroscope_data['y']
        gyro_z = gyroscope_data['z']
    else: return
    return [time.time(), acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

def start_activity(button):
    global measuring_started
    if int(button) == 1: #button pressed
        measuring_started = True
        
sensor.register_callback('button_1', start_activity)

# had to do it like this, because sensor data were freezed in the callback
while True:
    if(measuring_started):
        print("Started " + activity)
        df = pd.DataFrame(columns=COLUMNS)
        for i in range(0, MAX_DATA_POINTS):
            data_row = get_data()
            if(data_row):
                df.loc[len(df)] = data_row
            time.sleep(0.001)
        safe_to_csv(df)
        print("Data saved.")
        sys.exit()