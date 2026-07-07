from djitellopy import Tello
from time import sleep

try:
    tello = Tello()
    tello.connect()
    print(tello.get_battery())
    tello.takeoff()
    sleep(30)
    tello.land()
    print(tello.get_battery())
    tello.end()
except KeyboardInterrupt:
    tello.land()
    print(tello.get_battery())
    tello.end()
