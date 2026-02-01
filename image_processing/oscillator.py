import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# 2D wave equation (finite-difference) simulating ripples from water drops
# u_tt = c^2 * (u_xx + u_yy) with simple damping and absorbing boundary (sponge)

# Parameters
nx, ny = 200, 200
Lx, Ly = 100.0, 100.0
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing='xy')

c = 1.0                  # wave speed
courant = 0.5            # Courant number (<= 1/sqrt(2) for 2D stability)
dt = courant * min(dx, dy) / c
c2dt2 = (c * dt)**2

# damping and sponge (absorb at edges)
damp_global = 0.0005
sponge_width = int(min(nx, ny) * 0.12)
sponge = np.ones((ny, nx))
for i in range(ny):
    for j in range(nx):
        di = min(i, ny - 1 - i)
        dj = min(j, nx - 1 - j)
        dmin = min(di, dj)
        if dmin < sponge_width:
            # quadratic ramp to zero absorption at interior
            frac = (sponge_width - dmin) / sponge_width
            sponge[i, j] = 1.0 - 0.9 * (frac**2)

# fields
u_prev = np.zeros((ny, nx), dtype=float)  # u at t-dt
u = np.zeros((ny, nx), dtype=float)       # u at t
u_next = np.zeros((ny, nx), dtype=float)  # u at t+dt

# define drops: list of (time_step, x_pos, y_pos, amplitude, sigma)
drops = []
# drop at center at t=5
drops.append((5, 0.0, 0.0, 3.0, 2.0))
# another drop offset at t=30
#drops.append((30, 10.0, -8.0, 2.5, 1.8))
# multiple quick drops
#drops.append((60, -15.0, 12.0, 2.0, 1.5))

# convert drop positions to grid indices for convenience
def add_drop(u_field, x_pos, y_pos, amplitude=1.0, sigma=1.0):
    # add Gaussian displacement centered at (x_pos,y_pos)
    r2 = (X - x_pos)**2 + (Y - y_pos)**2
    u_field += amplitude * np.exp(-r2 / (2.0 * sigma**2))

# simulation steps
nsteps = 200
frames_to_record = list(range(0, nsteps, max(1, nsteps // 40)))

# For plotting limits, track max amplitude
expected_amp = max(d[3] for d in drops) if drops else 1.0
vmax = expected_amp * 0.9

# Prepare figure for animation
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(u, extent=[x.min(), x.max(), y.min(), y.max()], cmap='seismic',
               vmin=-vmax, vmax=vmax, origin='lower')
ax.set_title('2D water-drop ripple (u)')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Precompute drop schedule by step
drops_by_step = {}
for (tstep, xp, yp, amp, sig) in drops:
    drops_by_step.setdefault(tstep, []).append((xp, yp, amp, sig))

# Helper to compute Laplacian via finite differences (vectorized)
def laplacian(uarr):
    return (np.roll(uarr, 1, axis=0) + np.roll(uarr, -1, axis=0)
            + np.roll(uarr, 1, axis=1) + np.roll(uarr, -1, axis=1)
            - 4.0 * uarr) / (dx * dy)

# Animation update
frame_index = 0
recorded_frames = []

def update(frame):
    global u_prev, u, u_next, frame_index
    # time step index
    t = frame_index

    # apply drops at this time
    if t in drops_by_step:
        for (xp, yp, amp, sig) in drops_by_step[t]:
            add_drop(u, xp, yp, amplitude=amp, sigma=sig)

    # compute Laplacian
    lap = laplacian(u)

    # standard second-order scheme: u_next = 2*u - u_prev + c^2 dt^2 * lap
    u_next = 2.0 * u - u_prev + c2dt2 * lap

    # apply mild global damping and sponge to absorb at boundaries
    u_next *= (1.0 - damp_global)
    u_next *= sponge

    # shift fields
    u_prev, u = u, u_next

    # update image
    im.set_data(u)
    frame_index += 1
    return [im]

# create animation
anim = animation.FuncAnimation(fig, update, frames=nsteps, interval=30, blit=True)

# show animation
plt.show()

# Optionally save: uncomment to save animation as MP4 (requires ffmpeg)
# anim.save('ripples.mp4', dpi=150, fps=20)
