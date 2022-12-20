import pyautogui
from pynput.keyboard import Listener as K
from pynput.mouse import Listener as M

def write_in_log(data):
    with open('output.txt', 'a') as f:
        f.write(data+"\n")

# Record keyboard input
def on_press(key):
    write_in_log(f"KD[{key}]")

def on_release(key):
    write_in_log(f"KU[{key}]")

# Record hotkeys
def on_hotkey(key):
    write_in_log(f"HK[{key}]")

def on_move(x, y):
    write_in_log("MT[{0},{1}]".format(x, y))

def on_click(x, y, button, pressed):
    button = str(button).removeprefix('Button.')
    if pressed:
        write_in_log('CK[{0},{1},button="{2}"]'.format(x, y, button))

def on_scroll(x, y, dx, dy):
    write_in_log('SC[({0},{1})({2},{3})]'.format(x, y, dx, dy))
    
# Create and start the second listener
keyboard_listener = K(on_press=on_press, on_release=on_release)
keyboard_listener.start()

# Create and start the first listener
mouse_listener = M(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
mouse_listener.start()

keyboard_listener.join()
mouse_listener.join()






