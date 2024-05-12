import sys
import json
from threading import Thread
from time import sleep
from datetime import datetime
import signal

# those modules are imported dynamically during runtime
# they are imported only if the corresponding class is used
#import socket
#import serial
#import wiimote

class Sensor():
    # class variable that stores all instances of Sensor
    instances = []

    def __init__(self):
        # list of strings which represent capabilites, such as 'buttons' or 'accelerometer'
        self._capabilities = []
        # for each capability, store a list of callback functions
        self._callbacks = {}
        # for each capability, store the last value as an object
        self._data = {}
        self._receiving = False
        Sensor.instances.append(self)

    # stops the loop in _receive() and kills the thread
    # so the program can terminate smoothly
    def disconnect(self):
        self._receiving = False
        Sensor.instances.remove(self)
        if self._connection_thread:
            self._connection_thread.join()

    # runs as a thread
    # receives json formatted data from sensor,
    # stores it and notifies callbacks
    def _update(self, data):
        try:
            data_json = json.loads(data)
        except json.decoder.JSONDecodeError:
            # incomplete data
            return

        for key, value in data_json.items():
            self._add_capability(key)

            # do not notify callbacks on initialization
            if self._data[key] == []:
                self._data[key] = value
                continue

            # notify callbacks only if data has changed
            if self._data[key] != value:
                self._data[key] = value
                self._notify_callbacks(key)

    # checks if capability is available
    def has_capability(self, key):
        return key in self._capabilities

    def _add_capability(self, key):
        if not self.has_capability(key):
            self._capabilities.append(key)
            self._callbacks[key] = []
            self._data[key] = []

    # returns a list of all current capabilities
    def get_capabilities(self):
        return self._capabilities

    # get last value for specified capability
    def get_value(self, key):
        try:
            return self._data[key]
        except KeyError:
            # notification when trying to get values for a non-existent capability
            #raise KeyError(f'"{key}" is not a capability of this sensor.')
            return None

    # register a callback function for a change in specified capability
    def register_callback(self, key, func):
        self._add_capability(key)
        self._callbacks[key].append(func)

    # remove already registered callback function for specified capability
    def unregister_callback(self, key, func):
        if key in self._callbacks:
            self._callbacks[key].remove(func)
            return True
        else:
            # in case somebody wants to check if the callback was present before
            return False

    def _notify_callbacks(self, key):
        for func in self._callbacks[key]:
            func(self._data[key])

# sensor connected via WiFi/UDP
# initialized with a UDP port
# listens to all IPs by default
# requires the socket module
class SensorUDP(Sensor):
    def __init__(self, port, ip='0.0.0.0'):
        Sensor.__init__(self)
        self._ip = ip
        self._port = port
        self._connect()

    def _connect(self):
        import socket

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self._ip, self._port))
        self._connection_thread = Thread(target=self._receive)
        self._connection_thread.start()

    def _receive(self):
        self._receiving = True
        while self._receiving:
            data, addr = self._sock.recvfrom(1024)
            try:
                data_decoded = data.decode()
            except UnicodeDecodeError:
                continue
            self._update(data_decoded)

# sensor connected via serial connection (USB)
# initialized with a path to a TTY (e.g. /dev/ttyUSB0)
# default baudrate is 115200
# requires pyserial
class SensorSerial(Sensor):
    def __init__(self, tty, baudrate=115200):
        Sensor.__init__(self)
        self._tty = tty
        self._baudrate = baudrate
        self._connect()

    def _connect(self):
        import serial

        self._serial = serial.Serial(self._tty)
        self._serial.baudrate = self._baudrate
        self._connection_thread = Thread(target=self._receive)
        self._connection_thread.start()

    def _receive(self):
        self._receiving = True
        try:
            while self._receiving:
                data = self._serial.readline()
                try:
                    data_decoded = data.decode()
                except UnicodeDecodeError:
                    continue
                self._update(data)
        except:
            # connection lost, try again
            self._connect()

# uses a Nintendo Wiimote as a sensor (connected via Bluetooth)
# initialized with a Bluetooth address
# requires wiimote.py (https://github.com/RaphaelWimmer/wiimote.py)
# and pybluez
class SensorWiimote(Sensor):
    def __init__(self, btaddr):
        Sensor.__init__(self)
        self._btaddr = btaddr
        self._connect()

    def _connect(self):
        import wiimote

        self._wiimote = wiimote.connect(self._btaddr)
        self._connection_thread = Thread(target=self._receive)
        self._connection_thread.start()

    def _receive(self):
        self._receiving = True
        buttons = self._wiimote.buttons.BUTTONS.keys()
        while self._receiving:
            x = self._wiimote.accelerometer[0]
            y = self._wiimote.accelerometer[1]
            z = self._wiimote.accelerometer[2]
            data_string = f'{{"x":{x},"y":{y},"z":{z}}}'
            self._update('accelerometer', data_string)

            for button in buttons:
                state = int(self._wiimote.buttons[button])
                self._update(f'button_' + button.lower(), state)
            sleep(0.001)

    def _update(self, key, value):
        self._add_capability(key)
        
        # do not notify callbacks on initialization
        if self._data[key] == []:
            self._data[key] = value
            return

        # notify callbacks only if data has changed
        if self._data[key] != value:
            self._data[key] = value
            self._notify_callbacks(key)

# close the program softly when ctrl+c is pressed
def handle_interrupt_signal(signal, frame):
    for sensor in Sensor.instances:
        sensor.disconnect()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt_signal)
