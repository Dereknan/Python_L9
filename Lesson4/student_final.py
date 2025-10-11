import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import random

GRID_SIZE = 4
GOAL_STATE = GRID_SIZE**2 - 1  # bottom-right
TRAP_STATE = 5                 # a bomb square

ACTIONS   = [0, 1, 2, 3]                # up, right, down, left
DIRECTIONS = [(-1,0),(0,1),(1,0),(0,-1)]  # row/col movement

# Rewards
def build_rewards():
    rewards = np.zeros(GRID_SIZE*GRID_SIZE)
    rewards[GOAL_STATE] = 10.0
    rewards[TRAP_STATE] = -10.0
    return rewards

# Helpers for movement
def index_to_rowcol(state): return divmod(state, GRID_SIZE)
def rowcol_to_index(r,c):   return r*GRID_SIZE + c

def take_step(state, action):
    """Apply action to state, bounce off walls if necessary."""
    r, c = index_to_rowcol(state)
    dr, dc = DIRECTIONS[action]
    nr, nc = r+dr, c+dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
        return rowcol_to_index(nr,nc)
    return state

def run_q_learning(
    step_cost=0.0,
    episodes=60,
    alpha=0.1,   # learning rate
    gamma=0.95,  # discount factor
    eps=0.95,    # exploration probability
    eps_min=0.05,
    eps_decay=0.99
):
    q_table = np.zeros((GRID_SIZE*GRID_SIZE, len(ACTIONS)))
    rewards = build_rewards()
    lengths = []

    for ep in range(episodes):
        s = 0
        steps = 0
        done = False

        while not done and steps < 200:
            # Step 1: choose an action
            # if random < eps -> random action, else pick best from q_table
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = int(np.argmax(q_table[s]))

            # Step 2: move to next state
            s2 = take_step(s, a)

            # Step 3: calculate reward (include step cost unless GOAL/TRAP)
            if s2 in (GOAL_STATE, TRAP_STATE):
                r = rewards[s2]
            else:
                r = step_cost

            # Step 4: update Q-value
            best_future = np.max(q_table[s2])
            q_table[s,a] += alpha * (r + gamma*best_future - q_table[s,a])

            s = s2
            steps += 1
            if s in (GOAL_STATE, TRAP_STATE):
                done = True

        # Step 5: decay exploration rate
        eps = max(eps_min, eps * eps_decay)
        lengths.append(steps)

    return q_table, np.array(lengths)

def greedy_policy(q_table): 
    return np.argmax(q_table, axis=1)

# Visualization
def draw_grid(ax, title=""):
    bg = np.zeros((GRID_SIZE, GRID_SIZE))
    ax.imshow(bg, cmap="Greys", vmin=0, vmax=1)
    for i in range(GRID_SIZE+1):
        ax.axhline(i-0.5, color="black")
        ax.axvline(i-0.5, color="black")
    for s in range(GRID_SIZE*GRID_SIZE):
        r,c = index_to_rowcol(s)
        if s == GOAL_STATE:
            ax.text(c,r,"Goal",ha="center",va="center",fontsize=16)
        elif s == TRAP_STATE:
            ax.text(c,r,"BOMB",ha="center",va="center",fontsize=14)
        else:
            ax.text(c,r,str(s),ha="center",va="center",fontsize=8,color="gray")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)

def arrows_for_policy(ax, pi, title="Policy"):
    draw_grid(ax, title)
    for s,a in enumerate(pi):
        if s in (GOAL_STATE, TRAP_STATE): continue
        r,c = index_to_rowcol(s)
        if a == 0: dx, dy = 0, -0.3
        elif a == 1: dx, dy = 0.3, 0
        elif a == 2: dx, dy = 0, 0.3
        else: dx, dy = -0.3, 0
        ax.arrow(c,r,dx,dy,head_width=0.15,head_length=0.15,
                 fc="tab:blue",ec="tab:blue")

def generate_episode(pi, max_steps=60):
    s = 0
    path = [s]
    for _ in range(max_steps):
        a = int(pi[s])
        s = take_step(s, a)
        path.append(s)
        if s in (GOAL_STATE, TRAP_STATE):
            break
    return path


# ====== Run their code + visualize ======
q_none, lens_none = run_q_learning(step_cost=0.0)
q_pen,  lens_pen  = run_q_learning(step_cost=-1.0)

pi_none = greedy_policy(q_none)
pi_pen  = greedy_policy(q_pen)

# Animate
path_none = generate_episode(pi_none)
path_pen  = generate_episode(pi_pen)

fig, axes = plt.subplots(1,2,figsize=(8,4))
for ax, title in zip(axes,["No step penalty","Step penalty = -1"]):
    draw_grid(ax, title)

dots = []
for ax in axes:
    d, = ax.plot([], [], 'o', color='tab:red', markersize=12)
    dots.append(d)

def init():
    for d in dots: d.set_data([],[])
    return dots

def animate(t):
    if t < len(path_none):
        r,c = index_to_rowcol(path_none[t])
        dots[0].set_data([c],[r])
    if t < len(path_pen):
        r2,c2 = index_to_rowcol(path_pen[t])
        dots[1].set_data([c2],[r2])
    return dots

ani = animation.FuncAnimation(fig, animate, init_func=init,
    frames=max(len(path_none), len(path_pen)), interval=500, blit=True)
plt.show()
