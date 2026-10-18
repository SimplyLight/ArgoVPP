import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Copié des fonctions de la cellule 2 du notebook

def parse_points_from_text(text):
    pts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ',' in line:
            parts = line.split(',')
        else:
            parts = line.split()
        try:
            x = float(parts[0])
            y = float(parts[1])
            pts.append((x, y))
        except Exception:
            continue
    return np.array(pts)


def close_polygon(pts):
    if len(pts) == 0:
        return pts
    if not (np.allclose(pts[0], pts[-1])):
        pts = np.vstack([pts, pts[0]])
    return pts


def intersect_segment_with_y(p1, p2, y):
    (x1, y1) = p1
    (x2, y2) = p2
    if (y1 - y) * (y2 - y) > 0:
        return None
    if y2 == y1:
        return None
    t = (y - y1) / (y2 - y1)
    if 0.0 <= t <= 1.0:
        x = x1 + t * (x2 - x1)
        return x
    return None


def intersections_with_line_y(polygon, y):
    xs = []
    for i in range(len(polygon)-1):
        p1 = polygon[i]
        p2 = polygon[i+1]
        xi = intersect_segment_with_y(p1, p2, y)
        if xi is not None:
            xs.append(xi)
    return np.array(xs)


def compute_origin_from_root_mid(pts, y_root=0.0):
    pts_closed = close_polygon(pts)
    xs = intersections_with_line_y(pts_closed, y_root)
    if len(xs) >= 2:
        x_min = np.min(xs)
        x_max = np.max(xs)
        x0 = 0.5*(x_min + x_max)
        y0 = y_root
        return x0, y0, xs
    C = np.mean(pts, axis=0)
    return C[0], C[1], np.array([])


# Exemple avec le texte par défaut utilisé dans le notebook
paste_text = '''0.0, -0.5
0.2, -0.5
0.3, 0.0
0.2, 0.5
0.0, 0.5
-0.2, 0.5
-0.3, 0.0
-0.2, -0.5
0.0, -0.5'''

pts = parse_points_from_text(paste_text)
if pts is None or len(pts) == 0:
    print('Aucun point valide trouvé.')
else:
    pts = close_polygon(pts)
    x0, y0, xs = compute_origin_from_root_mid(pts, y_root=0.0)
    print('Intersections:', xs)
    print(f'Origine: x0={x0:.3f}, y0={y0:.3f}')
    # Trace et sauvegarde
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(pts[:,0], pts[:,1], '-k')
    ax.scatter(pts[:-1,0], pts[:-1,1], s=20, color='C0')
    ax.plot(x0, y0, 'rX', markersize=10, label='origine (x0,y0)')
    if len(xs) >= 2:
        ax.plot([np.min(xs), np.max(xs)], [0.0, 0.0], 'r--', label='côte racine')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title("Prévisualisation (script test)")
    ax.legend()
    out_png = 'c:\\Users\\leo.chassot\\Desktop\\moment aile microtransat\\test_cell2.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print('Image saved to', out_png)
