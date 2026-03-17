#!/usr/bin/env python3
"""
OJCOMS scenario figure: road network + 3 vehicle categories
Cooperative (commuter+fleet) / Random / Selfish
"""
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess, re

NET = "../simulations/manhattan/optics_valley.net.xml"
ROU = "../simulations/manhattan/optics_valley_commuter.rou.xml"
CFG = "../simulations/manhattan/optics_valley_commuter.sumocfg"

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# 1. Parse road network edges
tree = ET.parse(NET)
root = tree.getroot()
edges_geo = {}
for edge in root.findall('.//edge'):
    eid = edge.get('id', '')
    if eid.startswith(':'): continue
    for lane in edge.findall('lane'):
        shape = lane.get('shape', '')
        if shape:
            pts = []
            for pair in shape.strip().split():
                xy = pair.split(',')
                if len(xy) >= 2:
                    pts.append((float(xy[0]), float(xy[1])))
            if len(pts) >= 2:
                edges_geo[eid] = pts
            break

# 2. Parse vehicles: type + first edge -> starting position
rou_tree = ET.parse(ROU)
rou_root = rou_tree.getroot()

vehicles = []
for veh in rou_root.findall('.//vehicle'):
    vid = veh.get('id')
    vtype = veh.get('type', '')
    route = veh.find('route')
    if route is None: continue
    edge_list = route.get('edges', '').split()
    if not edge_list: continue
    
    # Get position from ~3rd edge (vehicles have moved a bit)
    for eidx in [min(3, len(edge_list)-1), 0]:
        first_edge = edge_list[eidx]
        # Try without prefix modifications
        for candidate in [first_edge, first_edge.lstrip('-')]:
            if candidate in edges_geo:
                pts = edges_geo[candidate]
                mid = len(pts) // 2
                x, y = pts[mid]
                # Classify into 3 categories
                if 'selfish' in vtype:
                    cat = 'Selfish'
                elif 'random' in vtype:
                    cat = 'Random'
                else:  # commuter_* or fleet_*
                    cat = 'Cooperative'
                vehicles.append((x, y, cat, vid))
                break
        else:
            continue
        break

print(f"Parsed {len(vehicles)} vehicles")
for cat in ['Cooperative', 'Random', 'Selfish']:
    n = sum(1 for v in vehicles if v[2] == cat)
    print(f"  {cat}: {n}")

# 3. Plot
fig, ax = plt.subplots(figsize=(8, 5.5))

# Road network (light gray)
for eid, pts in edges_geo.items():
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color='#BBBBBB', linewidth=0.6, alpha=0.8, zorder=1)

# Vehicles by category
style = {
    'Cooperative': {'color': '#0072B2', 'marker': 'o', 'size': 18, 'zorder': 3, 'label': 'Cooperative (commuter+fleet, 71%)'},
    'Random':      {'color': '#808080', 'marker': 'v', 'size': 18, 'zorder': 3, 'label': 'Random (19%)'},
    'Selfish':     {'color': '#D55E00', 'marker': 'X', 'size': 30, 'zorder': 5, 'label': 'Selfish (low trust, 10%)'},
}

for cat in ['Cooperative', 'Random', 'Selfish']:
    s = style[cat]
    vx = [v[0] for v in vehicles if v[2] == cat]
    vy = [v[1] for v in vehicles if v[2] == cat]
    ax.scatter(vx, vy, c=s['color'], marker=s['marker'], s=s['size'],
               zorder=s['zorder'], label=s['label'], alpha=0.8, edgecolors='white', linewidths=0.3)

# Road name annotations
all_x = [p[0] for pts in edges_geo.values() for p in pts]
all_y = [p[1] for pts in edges_geo.values() for p in pts]
xmin, xmax = min(all_x), max(all_x)
ymin, ymax = min(all_y), max(all_y)

# Approximate road label positions (from the thesis figure)
labels = [
    (1800, 2650, 'Luoyu Rd.', 0),
    (2800, 1950, 'Guanshan Ave.', 90),
    (1400, 1600, 'Xiongchu Ave.', 0),
]
for lx, ly, txt, rot in labels:
    ax.text(lx, ly, txt, fontsize=9, fontstyle='italic', fontweight='bold',
            rotation=rot, ha='center', va='center', color='#444444',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# Scale bar
sx, sy = xmin + 200, ymin + 100
ax.plot([sx, sx+1000], [sy, sy], 'k-', linewidth=2.5)
ax.text(sx+500, sy+60, '1 km', ha='center', fontsize=9, fontweight='bold')

# Legend
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9, edgecolor='#cccccc',
          markerscale=1.2)

ax.set_xlabel('X (m)', fontsize=11)
ax.set_ylabel('Y (m)', fontsize=11)
ax.set_title('Wuhan Optics Valley Simulation Scenario',
             fontsize=11, pad=8)
ax.set_aspect('equal')
ax.tick_params(labelsize=9)

# Crop to vehicle-dense area
vxs = [v[0] for v in vehicles]
vys = [v[1] for v in vehicles]
if vxs:
    pad = 300
    ax.set_xlim(min(vxs)-pad, max(vxs)+pad)
    ax.set_ylim(min(vys)-pad, max(vys)+pad)

OUT = "../simulations/manhattan/optics_valley_ojcoms.png"
fig.savefig(OUT)
fig.savefig(OUT.replace('.png', '.pdf'))
plt.close()
print(f"Saved: {OUT}")

# Copy to Downloads
import shutil
shutil.copy2(OUT, "/mnt/hgfs/Downloads/optics_valley_ojcoms.png")
shutil.copy2(OUT.replace('.png','.pdf'), "/mnt/hgfs/Downloads/optics_valley_ojcoms.pdf")
print("Copied to /mnt/hgfs/Downloads/")
