import numpy as np
import matplotlib.pyplot as plt

# initial 2D Gaussian wavepacket with plane-wave momentum
def gaussian_2d(x, y, x0=0.0, y0=0.0, sigma=1.0, kx0=5.0, ky0=0.0):
    X, Y = np.meshgrid(x, y, indexing='xy')
    envelope = np.exp(-((X - x0)**2 + (Y - y0)**2) / (4.0 * sigma**2))
    plane = np.exp(1j * (kx0 * X + ky0 * Y))
    return envelope * plane

# spatial grid
nx, ny = 256, 256
x = np.linspace(-20, 20, nx)
y = np.linspace(-20, 20, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing='xy')

# wavepacket parameters (units: m = 1, ħ = 1)
x0, y0 = 0.0, 0.0
kx0, ky0 = 0.0, 0.0
sigma = 1.0

# build and normalize initial wavefunction
psi0 = gaussian_2d(x, y, x0=x0, y0=y0, sigma=sigma, kx0=kx0, ky0=ky0)
prob0 = np.abs(psi0)**2
norm0 = np.sqrt(np.sum(prob0) * dx * dy)
psi0 /= norm0

# Fourier-space grid
kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='xy')
K2 = KX**2 + KY**2

# precompute initial FFT
Psi_k0 = np.fft.fft2(psi0)

# times to show (choose values where motion is visible)
times = [0.0, 1.0, 2.0, 4.0]

# plotting probability density at each time and mark expected centre
fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4))
extent = [x.min(), x.max(), y.min(), y.max()]
for ax, t in zip(axes, times):
    # free evolution in momentum space: Psi(k,t) = Psi(k,0) * exp(-i * (k^2 / (2m)) * t) with m=1
    phase = np.exp(-1j * (K2) * t / 2.0)
    Psi_k_t = Psi_k0 * phase
    psi_t = np.fft.ifft2(Psi_k_t)
    prob_t = np.abs(psi_t)**2

    im = ax.imshow(prob_t, extent=extent, origin='lower', cmap='viridis')
    ax.set_title(f't = {t:.2f}')
    ax.set_xlabel('x')
    if ax is axes[0]:
        ax.set_ylabel('y')

    # expected centre moves with group velocity v = k0 / m (m=1)
    cx = x0 + kx0 * t
    cy = y0 + ky0 * t
    ax.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# approximate energy from central momentum (m=1): E = k^2 / 2
k0_mag = np.sqrt(kx0**2 + ky0**2)
energy0 = 0.5 * k0_mag**2

# 3D surface plots of probability density at each time (downsampled for speed)
from mpl_toolkits.mplot3d import Axes3D
fig3 = plt.figure(figsize=(12, 8))
num = len(times)
cols = min(2, num)
rows = int(np.ceil(num / cols))
step = max(1, nx // 80)  # downsample factor for plotting
for i, t in enumerate(times):
    ax = fig3.add_subplot(rows, cols, i + 1, projection='3d')
    phase = np.exp(-1j * (K2) * t / 2.0)
    Psi_k_t = Psi_k0 * phase
    psi_t = np.fft.ifft2(Psi_k_t)
    prob_t = np.abs(psi_t)**2

    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Z = prob_t[::step, ::step]

    surf = ax.plot_surface(Xs, Ys, Z, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_title(f't={t:.2f}, E≈{energy0:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('|ψ|^2')
    fig3.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()

# Aggregate 2D overlay: contours of probability at each time and the centre trajectory
fig4, ax4 = plt.subplots(figsize=(7, 7))
colors = plt.cm.plasma(np.linspace(0, 1, len(times)))
contour_levels = [0.2, 0.4, 0.6, 0.8]  # fractions of peak for each time
centers_x = []
centers_y = []
for c, t in zip(colors, times):
    phase = np.exp(-1j * (K2) * t / 2.0)
    Psi_k_t = Psi_k0 * phase
    psi_t = np.fft.ifft2(Psi_k_t)
    prob_t = np.abs(psi_t)**2
    # normalize for contouring
    prob_norm = prob_t / prob_t.max()
    # plot contour lines at fractional levels
    cs = ax4.contour(X, Y, prob_norm, levels=contour_levels, colors=[c], linewidths=1.0, alpha=0.9)
    # mark the peak centre
    cx = x0 + kx0 * t
    cy = y0 + ky0 * t
    centers_x.append(cx)
    centers_y.append(cy)
    ax4.plot(cx, cy, marker='+', color=c, markersize=10, markeredgewidth=2)
