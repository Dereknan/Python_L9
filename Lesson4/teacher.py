# 4x4 Gridworld (Tkinter) + Q-learning
# ------------------------------------
# Keys for students:
# - Agent learns by trial & error to reach 🏆 and avoid 💣
# - Small step penalty (-1) encourages faster paths
# - Q-table stores "how good" each action is in each state

import tkinter as tk
import numpy as np
import random
import time

# ====== World config ======
N = 4                             # grid size (N x N)
START = 0                         # starting at top-left (state index)
GOAL  = N*N - 1                   # reward at bottom-right
TRAP  = 5                         # change this to move the bomb 💣

STEP_PENALTY = -1.0               # "living cost" each move
REWARD_GOAL  = +10.0
REWARD_TRAP  = -10.0

# Actions: 0=up, 1=right, 2=down, 3=left
ACTIONS = [0, 1, 2, 3]
DIRS    = [(-1,0), (0,1), (1,0), (0,-1)]

# ====== Learning hyperparameters ======
alpha   = 0.1     # learning rate (Like gradient descent)
gamma   = 0.95    # discount factor (Consideration for the future)
epsilon = 0.9     # how much to explore
eps_min = 0.05    # still explore 5% of the time
eps_decay = 0.97  # decay per episode (Explore less)
episodes = 60     # keep short so they can watch it learn

# ====== Q-table ======
q = np.zeros((N*N, len(ACTIONS)))  # rows=states, cols=actions

# ----- Helpers: state/index conversions -----
def rc_to_s(r, c): return r * N + c
def s_to_rc(s):    return divmod(s, N)

# ----- Step function (environment) -----
def step(state, action):
    r, c = s_to_rc(state)
    dr, dc = DIRS[action]
    nr, nc = r + dr, c + dc

    # hit wall? stay in place
    if not (0 <= nr < N and 0 <= nc < N):
        nr, nc = r, c

    new_state = rc_to_s(nr, nc)

    # reward logic
    if new_state == GOAL:
        reward = REWARD_GOAL
    elif new_state == TRAP:
        reward = REWARD_TRAP
    else:
        reward = STEP_PENALTY  # small cost per move

    done = (new_state == GOAL) or (new_state == TRAP)
    return new_state, reward, done

# ====== Tkinter UI ======
CELL = 70
PAD  = 16

root = tk.Tk()
root.title("4x4 Gridworld — Q-learning")

canvas = tk.Canvas(root, width=N*CELL, height=N*CELL)
canvas.pack()

info = tk.Label(root, text="Episode: 0   Steps: 0   Epsilon: 0.00")
info.pack(pady=6)

def draw_grid(agent_state):
    canvas.delete("all")

    # grid lines + cell labels
    for r in range(N):
        for c in range(N):
            x1 = c*CELL; y1 = r*CELL
            x2 = x1+CELL; y2 = y1+CELL
            canvas.create_rectangle(x1, y1, x2, y2, outline="black")

            s = rc_to_s(r, c)
            if s == GOAL:
                canvas.create_text(x1+CELL/2, y1+CELL/2, text="🏆", font=("Arial", 24))
            elif s == TRAP:
                canvas.create_text(x1+CELL/2, y1+CELL/2, text="💣", font=("Arial", 24))
            else:
                canvas.create_text(x1+CELL-12, y1+14, text=str(s), fill="gray")

    # draw agent as a red circle
    ar, ac = s_to_rc(agent_state)
    cx = ac*CELL + CELL/2
    cy = ar*CELL + CELL/2
    canvas.create_oval(cx-PAD, cy-PAD, cx+PAD, cy+PAD, fill="tomato", outline="black")

# ====== Training (animated) ======
def run_training():
    global epsilon, episode
    episode = 0
    def run_episode():
        global episode
        if episode >= episodes:
            info.config(text="Training done! Try changing TRAP/STEP_PENALTY and rerun.")
            return

        steps = 0
        state = START
        done = False

        def step_once():
            nonlocal state, steps, done

            # render
            draw_grid(state)
            info.config(text=f"Episode: {episode+1}/{episodes}   Steps: {steps}   Epsilon: {epsilon:.2f}")
            root.update()

            # choose action (ε-greedy)
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = int(np.argmax(q[state]))

            # environment transition
            new_state, reward, done = step(state, action)

            # Q-learning update
            q[state, action] += alpha * (
                reward + gamma * np.max(q[new_state]) - q[state, action]
            )

            state = new_state
            steps += 1

            if done or steps > 200:
                # episode finished
                time.sleep(0.2)
                return after_episode()
            else:
                # continue episode after a short delay (animation speed)
                root.after(150, step_once)

        def after_episode():
            global episode
            # decay epsilon each episode (less random over time)
            global epsilon
            epsilon = max(eps_min, epsilon * eps_decay)
            episode += 1
            # start next episode after a tiny pause
            root.after(300, run_episode)

        # kick off this episode
        step_once()

    run_episode()

# Button to (re)start
tk.Button(root, text="Start / Restart", command=run_training).pack(pady=6)

# Tip label for students
tk.Label(root, text="Try: change TRAP, STEP_PENALTY, episodes, or move GOAL.").pack(pady=4)

draw_grid(START)
root.mainloop()
