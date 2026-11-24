import cv2

_plt = None
_fig = None
_ax = None
_im = None


def _ensure_backend():
    global _plt
    if _plt is not None:
        return _plt
    import matplotlib

    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

    import matplotlib.pyplot as plt

    plt.ion()
    _plt = plt
    return plt


def show_image(img_bgr, title=""):
    global _fig, _ax, _im

    plt = _ensure_backend()
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
    if _plt is None or _fig is None:
        return
    if _plt.fignum_exists(_fig.number):
        _plt.pause(0.001)


def close_display():
    global _fig, _ax, _im

    if _plt is not None and _fig is not None and _plt.fignum_exists(_fig.number):
        _plt.close(_fig)

    _fig = None
    _ax = None
    _im = None
