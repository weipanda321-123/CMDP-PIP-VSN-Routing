#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NET_FILE = "../simulations/manhattan/optics_valley.net.xml"

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

tree = ET.parse(NET_FILE)
root = tree.getroot()

edges = []
for edge in root.findall('.//edge'):
    if edge.get('id', '').startswith(':'): continue
    for lane in edge.findall('lane'):
        shape = lane.get('shape', '')
        if shape:
            pts = []
            for pair in shape.strip().split():
                xy = pair.split(',')
                if len(xy) >= 2: pts.append((float(xy[0]), float(xy[1])))
            if len(pts) >= 2: edges.append(pts)
            break

all_x = [p[0] for e in edges for p in e]
all_y = [p[1] for e in edges for p in e]
xmin, xmax = min(all_x), max(all_x)
ymin, ymax = min(all_y), max(all_y)

fig, ax = plt.subplots(figsize=(7, 5))
for pts in edges:
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color='#333333', linewidth=0.5, alpha=0.7)

sx, sy = xmin + 200, ymin + 150
ax.plot([sx, sx+1000], [sy, sy], 'k-', linewidth=2)
ax.text(sx+500, sy+80, '1 km', ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Wuhan Optics Valley Road Network (OpenStreetMap)', fontsize=12)
ax.set_aspect('equal')
w = (xmax-xmin)/1000; h = (ymax-ymin)/1000
ax.text(0.02, 0.02, f'{w:.1f} km x {h:.1f} km\n1,175 vehicles',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.savefig('../simulations/manhattan/optics_valley_network.png')
fig.savefig('../simulations/manhattan/optics_valley_network.pdf')
print(f"OK: {len(edges)} edges, {w:.1f}km x {h:.1f}km")

# 也复制到Downloads共享
import shutil
for ext in ['png', 'pdf']:
    src = f'../simulations/manhattan/optics_valley_network.{ext}'
    dst = f'/home/wei/Downloads/optics_valley_network.{ext}'
    shutil.copy2(src, dst)
    print(f"Copied to {dst}")
plt.close()
