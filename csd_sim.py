import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

N = 120
steps = 360
dt = 0.05
dx = 1.0

D = 1.0
a = 0.15
epsilon = 0.02
gamma = 0.5

def laplacian(grid):
    return (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid
    ) / (dx**2)

# Zorg dat de doelmap bestaat (Slidev public map)
output_dir = "slides/public"
os.makedirs(output_dir, exist_ok=True)

def generate_mp4(name, u0, cmap):
    print(f"Genereren: {name}.mp4 ...")
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    
    center = N // 2
    radius = 7
    y, x = np.ogrid[:N, :N]
    u[(x - center)**2 + (y - center)**2 <= radius**2] = u0

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor('#0f172a')
    ax.axis('off')
    im = ax.imshow(u, cmap=cmap, vmin=0, vmax=0.9, origin='lower')
    plt.tight_layout()

    def update(frame):
        nonlocal u, v
        for _ in range(3):
            lap_u = laplacian(u)
            du = D * lap_u + u * (1.0 - u) * (u - a) - v
            dv = epsilon * (u - gamma * v)
            u += du * dt
            v += dv * dt
        im.set_array(u)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=steps // 3, interval=33, blit=True)
    out_path = os.path.join(output_dir, f"{name}.mp4")
    ani.save(out_path, writer="ffmpeg", fps=30)
    plt.close()
    print(f"Opgeslagen in {out_path}")

# Run 1: Sub-drempel (dooft uit)
generate_mp4("csd_recovery", 0.10, "viridis")

# Run 2: Boven drempel (CSD golf)
generate_mp4("csd_wave", 0.85, "magma")