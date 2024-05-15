# this program visualizes activities with pyglet

import activity_recognizer as activity
import pyglet
from DIPPID import SensorUDP
import time
import pandas as pd

SLEEP_TIME = 0.01
MAX_DATA_POINTS = 500 #should be around 5 seconds

COLUMNS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

PORT = 5700
sensor = SensorUDP(PORT)

activity_recognizer = activity.ActivityRecognizer()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
pyglet.gl.glClearColor(1, 1, 1, 1) #make background white
TARGET_IMAGE_HEIGHT = 250

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

def measure() -> pd.DataFrame:
    df = pd.DataFrame(columns=COLUMNS)
    for i in range(0, MAX_DATA_POINTS):
        data_row = get_data()
        if(data_row):
            df.loc[len(df)] = data_row
        time.sleep(SLEEP_TIME)
    return df

#https://pyglet.readthedocs.io/en/latest/programming_guide/image.html
jumpingjack_img_1 = pyglet.image.load('img/jumpingjack_1.png')
jumpingjack_img_2 = pyglet.image.load('img/jumpingjack_2.png')
lifting_img_1 = pyglet.image.load('img/lifting_1.png')
lifting_img_2 = pyglet.image.load('img/lifting_2.png')
rowing_img_1 = pyglet.image.load('img/rowing_1.png')
rowing_img_2 = pyglet.image.load('img/rowing_2.png')
running_img_1 = pyglet.image.load('img/running_1.png')
running_img_2 = pyglet.image.load('img/running_2.png')

def draw_text(current_activity) -> None:
    text = "You are currently " + current_activity[0] + "!"

    activity_label = pyglet.text.HTMLLabel(str(text),
                                           x=window.width // 2, y=window.height - 50,
                                           anchor_x='center', anchor_y='center')
    activity_label.set_style('color', (0, 0, 0, 255))
    activity_label.set_style('font_name', 'Courier New')
    activity_label.set_style('font_size', 30) 
    activity_label.set_style('bold', True)

    activity_label.draw()

# get images based on activities
def get_images(current_activity) -> tuple:
    img_1 = None
    img_2 = None
    if current_activity == "jumpingjacks":
        img_1 = jumpingjack_img_1
        img_2 = jumpingjack_img_2
    elif current_activity == "lifting":
        img_1 = lifting_img_1
        img_2 = lifting_img_2
    elif current_activity == "rowing":
        img_1 = rowing_img_1
        img_2 = rowing_img_2
    elif current_activity == "running":
        img_1 = running_img_1
        img_2 = running_img_2
    return img_1, img_2

def draw_sprites(img_1, img_2) -> None:
    sprite_1 = pyglet.sprite.Sprite(img=img_1)
    sprite_2 = pyglet.sprite.Sprite(img=img_2)

    #scale images
    scale_factor = TARGET_IMAGE_HEIGHT / sprite_1.height
    sprite_1.width = sprite_1.width * scale_factor
    sprite_1.height = sprite_1.height * scale_factor
    scale_factor = TARGET_IMAGE_HEIGHT / sprite_2.height
    sprite_2.width = sprite_2.width * scale_factor
    sprite_2.height = sprite_2.height * scale_factor
    
    # calculate center position per half - used GPT for this
    if sprite_1 is not None and sprite_2 is not None:
        left_image_x = (window.width - 2 * sprite_1.width) // 4
        right_image_x = left_image_x + (window.width // 2)
        image_y = (window.height - sprite_1.height) // 2
        
        sprite_1.x = left_image_x
        sprite_2.x = right_image_x

        sprite_1.y = image_y
        sprite_2.y = image_y

        sprite_1.draw()
        sprite_2.draw()

# show UI
def show_activity(current_activity) -> None:
    draw_text(current_activity)
    img_1, img_2 = get_images(current_activity)
    if img_1 is not None and img_2 is not None:
        draw_sprites(img_1, img_2)
    
@window.event
def on_draw():
    df = measure()
    current_activity = activity_recognizer.get_current_activity(df)
    print(current_activity[0])
    window.clear()
    show_activity(current_activity)

pyglet.app.run() 