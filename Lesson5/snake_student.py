import numpy as np
import random
import tkinter as tk
import time

# -------------------------
# World / Actions
# -------------------------
GRID = 5
ACTIONS = [0, 1, 2, 3]          # 0=up, 1=right, 2=down, 3=left
DIRS = [(-1,0), (0,1), (1,0), (0,-1)]

# Rewards
REWARD_FOOD  = +10
REWARD_CRASH = -10
STEP_COST    = -1               # small step penalty to encourage speed

# Training hyperparams
EPISODES  = 200
ALPHA     = 0.1                 # learning rate
GAMMA     = 0.95                # discount factor
EPS_START = 0.9                 # exploration prob (ε)
EPS_MIN   = 0.05
EPS_DECAY = 0.99
MAX_STEPS = 60                  # safety cap per episode

# NEW: training stall detector (end episode if no progress)
STALL_LIMIT = 20  # steps without getting closer to food (restart episode)

# Demo (GUI) time limit
EPISODE_TIME_LIMIT_MS = 4000  # 4 seconds per demo episode

# -------------------------
# Env helpers (provided)
# -------------------------
def place_food():
    return random.randint(0, GRID*GRID - 1)

def s_to_rc(s): return divmod(s, GRID)
def rc_to_s(r, c): return r*GRID + c

def manhattan_dist(a, b):
    ar, ac = s_to_rc(a)
    br, bc = s_to_rc(b)
    return abs(ar - br) + abs(ac - bc)

def step(state, action, food):
    """One move of the snake head. Returns (new_state, reward, done)."""
    r, c = s_to_rc(state)
    dr, dc = DIRS[action]
    nr, nc = r + dr, c + dc
    # hit wall?
    if not (0 <= nr < GRID and 0 <= nc < GRID):
        return state, REWARD_CRASH, True
    new_state = rc_to_s(nr, nc)
    if new_state == food:
        return new_state, REWARD_FOOD, True  # ate food → episode ends (simple version)
    return new_state, STEP_COST, False

# -------------------------
# Q-learning (FILL ME)
# -------------------------
def run_q_learning(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
                   eps=EPS_START, eps_min=EPS_MIN, eps_decay=EPS_DECAY):
    q = np.zeros((GRID*GRID, len(ACTIONS)), dtype=np.float32)
    lengths = []

    for ep in range(episodes):
        s = random.randint(0, GRID*GRID - 1)   # random start
        food = place_food()
        done = False
        steps = 0

        # stall detector: track progress toward food
        prev_dist = manhattan_dist(s, food)
        stall_steps = 0

        while not done and steps < MAX_STEPS:
            # TODO 1: ε-greedy action selection
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = int(np.argmax(q[s]))

            # TODO 2: take one env step
            s2, r, done = step(s, a, food)

            # TODO 3: Q-learning update
            q[s, a] += alpha * (r + gamma * np.max(q[s2]) - q[s, a])

            # stall detection (no progress toward food)
            curr_dist = manhattan_dist(s2, food)
            if curr_dist < prev_dist:
                stall_steps = 0
            else:
                stall_steps += 1
            prev_dist = curr_dist
            if stall_steps >= STALL_LIMIT:
                # treat as terminal (timeout)
                done = True

            s = s2
            steps += 1

        eps = max(eps_min, eps * eps_decay)
        lengths.append(steps)

    return q, lengths

# -------------------------
# Visualization (provided & safe)
# -------------------------
CELL = 64
root = tk.Tk()
root.title("Mini-Snake (Student)")

CLOSED = False
def on_close():
    global CLOSED
    CLOSED = True
    try:
        root.destroy()
    except Exception:
        pass

root.protocol("WM_DELETE_WINDOW", on_close)

canvas = tk.Canvas(root, width=GRID*CELL, height=GRID*CELL)
canvas.pack()

def draw(state, food, title=""):
    if CLOSED or not root.winfo_exists() or not canvas.winfo_exists():
        return False
    try:
        canvas.delete("all")
        # grid
        for r in range(GRID):
            for c in range(GRID):
                x1, y1 = c*CELL, r*CELL
                canvas.create_rectangle(x1, y1, x1+CELL, y1+CELL, outline="black")
        # food
        fr, fc = s_to_rc(food)
        canvas.create_text(fc*CELL + CELL//2, fr*CELL + CELL//2, text="🍏", font=("Arial", 22))
        # snake head
        sr, sc = s_to_rc(state)
        pad = 10
        canvas.create_oval(sc*CELL+pad, sr*CELL+pad, sc*CELL+CELL-pad, sr*CELL+CELL-pad,
                           fill="seagreen")
        if title:
            canvas.create_text(10, 10, text=title, anchor="nw", font=("Arial", 12))
        root.update_idletasks()
        root.update()
        return True
    except tk.TclError:
        return False

def greedy_policy(q): return np.argmax(q, axis=1)

def demo_greedy(q, runs=3, delay_ms=120):
    pi = greedy_policy(q)
    for i in range(runs):
        if CLOSED: break
        s = random.randint(0, GRID*GRID - 1)
        food = place_food()
        done = False
        steps = 0
        episode_start = time.monotonic()

        while not done and steps < MAX_STEPS and not CLOSED:
            # time cutoff for episode
            elapsed_ms = (time.monotonic() - episode_start) * 1000.0
            if elapsed_ms >= EPISODE_TIME_LIMIT_MS:
                break

            if not draw(s, food, title=f"Greedy demo run {i+1}"):
                return
            a = int(pi[s])
            s, _, done = step(s, a, food)
            steps += 1
            root.after(delay_ms)
            if CLOSED: return
        root.after(250)

if __name__ == "__main__":
    q, lengths = run_q_learning()
    demo_greedy(q)
    root.mainloop()
