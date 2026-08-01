import csv
import numpy as np

INPUT  = '/sim_ws/src/pure_pursuit/waypoints/waypoints.csv'
OUTPUT = '/sim_ws/src/pure_pursuit/waypoints/waypoints_speed.csv'

SPEED_MAX = 6.0  #5.2
SPEED_MIN = 2.0  #1.8
SMOOTH_WINDOW = 5

def compute_curvature(pts):
    dx  = np.gradient(pts[:, 0])
    dy  = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    return np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2 + 1e-6)**1.5

def smooth(arr, window):
    return np.convolve(arr, np.ones(window)/window, mode='same')

pts = []
with open(INPUT) as f:
    for row in csv.DictReader(f):
        pts.append([float(row['x']), float(row['y'])])

pts = np.array(pts)
curv = smooth(compute_curvature(pts), SMOOTH_WINDOW)
curv_norm = (curv - curv.min()) / (curv.max() - curv.min() + 1e-6)
speeds = smooth(SPEED_MAX - curv_norm * (SPEED_MAX - SPEED_MIN), SMOOTH_WINDOW)

with open(OUTPUT, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y', 'speed'])
    for (x, y), s in zip(pts, speeds):
        writer.writerow([f'{x:.4f}', f'{y:.4f}', f'{s:.4f}'])

print(f'Saved {len(pts)} waypoints | speed {speeds.min():.2f}–{speeds.max():.2f} m/s')
