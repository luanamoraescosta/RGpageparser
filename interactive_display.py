import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import cv2
import numpy as np

plt.ion()

_fig = None
_ax = None
_im = None


def show_image(img_bgr, title=""):
    global _fig, _ax, _im

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if _fig is None or not plt.fignum_exists(_fig.number):
        _fig, _ax = plt.subplots(figsize=(13, 8))
        _ax.axis("off")
        _im = _ax.imshow(rgb)
    else:
        if _im.get_array().shape != rgb.shape:
            _ax.clear()
            _ax.axis("off")
            _im = _ax.imshow(rgb)
        else:
            _im.set_data(rgb)

    if title:
        try:
            _fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

    _fig.canvas.draw_idle()
    plt.pause(0.05)


def pump_events():
    if _fig is not None and plt.fignum_exists(_fig.number):
        plt.pause(0.001)


def close_display():
    global _fig, _ax, _im

    if _fig is not None and plt.fignum_exists(_fig.number):
        plt.close(_fig)

    _fig = None
    _ax = None
    _im = None
