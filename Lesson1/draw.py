# ==============================
# 10x10 Emoji Painter + Predictor (PyTorch)
# ==============================

# ---- Imports ----
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Constants ----
GRID = 10            # 10x10 pixels
CKPT_PATH = "smilesad.pth"   # saved model file (state_dict)


# ---- Model Definition ----
class EmojiMLP(nn.Module):
    """Tiny MLP: 100 -> 64 -> 32 -> 2"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # outputs: sad(0), happy(1)

    def forward(self, x):
        x = F.relu(self.fc1(x))      # hidden layer 1
        x = F.relu(self.fc2(x))      # hidden layer 2
        x = self.fc3(x)              # logits (no softmax here)
        return x


# ---- Load Trained Model ----
def load_model(path: str) -> nn.Module:
    """Load weights into the same model structure."""
    model = EmojiMLP()
    state = torch.load(path, map_location="cpu")
    # If you saved only state_dict -> load it directly
    if isinstance(state, dict) and all(k.startswith(("fc1", "fc2", "fc3")) for k in state.keys()):
        model.load_state_dict(state)
    else:
        # If you saved a dict with "model_state"
        model.load_state_dict(state.get("model_state", state))
    model.eval()
    return model


# Try to load; give a helpful message if not found
try:
    model = load_model(CKPT_PATH)
    print("Model loaded and ready.")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find '{CKPT_PATH}'. Train your model and save it as this filename first."
    )


# ---- Canvas (the 10x10 drawing) ----
img = np.zeros((GRID, GRID), dtype=np.float32)  # students will click to toggle 0/1


# ---- Templates (optional helpers) ----
def set_happy(canvas: np.ndarray) -> None:
    """Fill the canvas with a simple happy face."""
    canvas[:] = 0
    canvas[3, 3] = 1; canvas[3, 6] = 1        # eyes
    for x in [3, 4, 5, 6]: canvas[7, x] = 1   # mouth base
    canvas[6, 3] = 1; canvas[6, 6] = 1        # mouth corners


def set_sad(canvas: np.ndarray) -> None:
    """Fill the canvas with a simple sad face."""
    canvas[:] = 0
    canvas[3, 3] = 1; canvas[3, 6] = 1        # eyes
    for x in [3, 4, 5, 6]: canvas[6, x] = 1   # mouth base (higher)
    canvas[7, 3] = 1; canvas[7, 6] = 1        # mouth corners


# ---- Prediction Helpers ----
def predict_probs(canvas: np.ndarray) -> np.ndarray:
    """
    Return probabilities [p(sad), p(happy)] from the PyTorch model.
    Expects a 10x10 canvas with values 0/1.
    """
    vec = torch.tensor(canvas.reshape(1, -1), dtype=torch.float32)  # shape: (1, 100)
    with torch.no_grad():
        logits = model(vec)                     # (1, 2)
        probs = F.softmax(logits, dim=1)[0]     # (2,)
    return probs.cpu().numpy()


def predict_label(canvas: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Return (label_text, confidence, probs_array)."""
    probs = predict_probs(canvas)
    idx = int(np.argmax(probs))
    label = "🙂 Happy" if idx == 1 else "🙁 Sad"
    conf = float(probs[idx])
    return label, conf, probs


# ---- UI: Matplotlib Figure + Buttons ----
fig, ax = plt.subplots(figsize=(5, 5))
plt.subplots_adjust(bottom=0.22)  # space for buttons

# Show the 10x10 grid
im = ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
ax.set_xticks(range(GRID)); ax.set_yticks(range(GRID))
ax.set_xticklabels([]); ax.set_yticklabels([])   # hide tick labels
ax.grid(True, which='both', linewidth=0.5)       # show grid lines
# ax.set_title("Draw your emoji: click squares to toggle")

# Live prediction text
txt = ax.text(0.02, 1.02, "Prediction: —", transform=ax.transAxes)

# Button areas
ax_clear = plt.axes([0.08, 0.05, 0.15, 0.07])
ax_happy = plt.axes([0.28, 0.05, 0.18, 0.07])
ax_sad   = plt.axes([0.50, 0.05, 0.18, 0.07])
ax_pred  = plt.axes([0.73, 0.05, 0.19, 0.07])

# Buttons
btn_clear = Button(ax_clear, 'Clear')
btn_happy = Button(ax_happy, 'Happy template')
btn_sad   = Button(ax_sad,   'Sad template')
btn_pred  = Button(ax_pred,  'Predict')


# ---- Event Handlers ----
def onclick(event):
    """Toggle a pixel between 0 and 1 when the student clicks."""
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    x, y = int(round(event.xdata)), int(round(event.ydata))
    if 0 <= x < GRID and 0 <= y < GRID:
        img[y, x] = 1.0 - img[y, x]  # toggle 0 ↔ 1
        im.set_data(img)
        fig.canvas.draw_idle()


def do_clear(event):
    """Clear the canvas."""
    img[:] = 0
    im.set_data(img)
    txt.set_text("Prediction: —")
    fig.canvas.draw_idle()


def do_happy(event):
    """Fill with a happy template."""
    set_happy(img)
    im.set_data(img)
    txt.set_text("Prediction: —")
    fig.canvas.draw_idle()


def do_sad(event):
    """Fill with a sad template."""
    set_sad(img)
    im.set_data(img)
    txt.set_text("Prediction: —")
    fig.canvas.draw_idle()


def do_predict(event):
    """Run the model and display the prediction + confidence."""
    try:
        label, conf, _ = predict_label(img)
        txt.set_text(f"Prediction: {label}  (conf {conf:.2f})")
    except Exception as e:
        txt.set_text(f"Prediction error: {e}")
    fig.canvas.draw_idle()


# ---- Wire up UI ----
cid = fig.canvas.mpl_connect('button_press_event', onclick)
btn_clear.on_clicked(do_clear)
btn_happy.on_clicked(do_happy)
btn_sad.on_clicked(do_sad)
btn_pred.on_clicked(do_predict)

plt.show()
