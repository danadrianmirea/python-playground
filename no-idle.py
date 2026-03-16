import ctypes
import time

# Windows API constants
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT)
    ]

SendInput = ctypes.windll.user32.SendInput

def move_mouse(dx, dy):
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
    inp = INPUT(INPUT_MOUSE, mi)
    SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

try:
    while True:
        move_mouse(-10, 0)   # move left
        time.sleep(10)

        move_mouse(10, 0)    # move right
        time.sleep(10)

except KeyboardInterrupt:
    print("Stopped.")