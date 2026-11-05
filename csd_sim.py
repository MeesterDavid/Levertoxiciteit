import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# 1. Grid & Simulatiesnelheid
N = 100              
dx = 1.0

# 2. Modelparameters (exact gelijk)
D = 2.5              
a = 0.12             # Drempelwaarde
epsilon = 0.008      
gamma = 0.5

def laplacian(grid):
    return (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid
    ) / (dx**2)

output_dir = "slides/public"
os.makedirs(output_dir, exist_ok=True)

def generate_mp4(filename, u0, radius, substeps, dt_sim, frames_count):
    print(f"Genereren: {filename} (u₀ = {u0}, r = {radius})...")
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    
    # Initiële schijf in het centrum (beide felgeel!)
    center = N // 2
    y, x = np.ogrid[:N, :N]
    mask = (x - center)**2 + (y - center)**2 <= radius**2
    u[mask] = u0

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor('#000000')
    ax.axis('off')
    
    # Magma colormap met vaste schaal: 0 = zwart, 1 = felgeel/wit
    im = ax.imshow(u, cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def update(frame):
        nonlocal u, v
        for _ in range(substeps):
            lap_u = laplacian(u)
            du = D * lap_u + u * (1.0 - u) * (u - a) - v
            dv = epsilon * (u - gamma * v)
            u += du * dt_sim
            v += dv * dt_sim
            
            u = np.clip(u, 0.0, 1.1)
            v = np.clip(v, 0.0, 1.1)
            
        im.set_array(u)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames_count, interval=33, blit=True)
    out_path = os.path.join(output_dir, filename)
    ani.save(out_path, writer="ffmpeg", fps=30)
    plt.close()
    print(f"Gereed: {out_path}")

# Run 1: Recovery (Sub-kritische straal r=3, vertraagd over 8s)
# Begint net zo felgeel (u0=0.95), maar vloeit in ~8 seconden langzaam weg naar zwart
generate_mp4(
    filename="csd_recovery.mp4",
    u0=0.95,
    radius=3,
    substeps=2,
    dt_sim=0.02,
    frames_count=240
)

# Run 2: Wave (Supra-kritische straal r=6)
# Begint felgeel en blaast op tot een grote ringgolf over 8 seconden
generate_mp4(
    filename="csd_wave.mp4",
    u0=0.95,
    radius=6,
    substeps=5,
    dt_sim=0.04,
    frames_count=240
)