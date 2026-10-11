#!/usr/bin/env python3
"""
Tabletop Traverse - core library (scan -> hazard map -> route -> MuJoCo rover run)

Used by rover_sim.py (command line) and server.py (web app).

Original CLI notes:
Tabletop Traverse - MuJoCo rover simulation
===========================================

Turns a LiDAR scan of a tabletop (or a built-in sample table) into a MuJoCo
heightfield, plans a safe route from A to B with the same hazard classifier and
A* planner as the Tabletop Traverse web app, and drives a 4-wheel skid-steer
rover along it with a pure-pursuit controller.

Outputs
  * an MP4 with an overview camera and a chase camera side by side
  * a JSON log: planned vs. driven distance, time, max tilt, tracking error

Examples
  python rover_sim.py                                   # sample table -> rover_run.mp4
  python rover_sim.py --scan table.ply --a 5 50 --b 70 8
  python rover_sim.py --view                            # live viewer (macOS: use `mjpython`)

Coordinates for --a / --b are in centimeters from the table's lower-left corner,
the same x / y numbers the web app shows when you hover the map.
"""
from __future__ import annotations

import heapq
import math
import os

import numpy as np
from scipy import ndimage

VOID, FREE, CLIMB, OBST, MARGIN = 0, 1, 2, 3, 4
CLASS_NAMES = ["off surface", "open", "small object", "obstacle", "keep-out margin"]
FLAT, ROUGH = 0.004, 0.003          # m: below this a cell counts as flat table
TABLE_Z = 0.04                      # m: table top sits this high in the sim, so edges are real drops

# hazard-map colours (same palette as the web app, light theme)
CLASS_RGB = np.array([[43, 52, 62], [220, 228, 216], [227, 172, 60], [197, 64, 44], [237, 198, 188]], np.uint8)


# --------------------------------------------------------------------------- scan loading
def load_ply(src) -> np.ndarray:
    """src: a file path or the raw bytes of a PLY file."""
    if isinstance(src, (bytes, bytearray)):
        buf = bytes(src)
    else:
        with open(src, "rb") as f:
            buf = f.read()
    end = buf.find(b"end_header")
    if not buf.startswith(b"ply") or end < 0:
        raise ValueError("Not a PLY file with a readable header.")
    off = end + len(b"end_header")
    if buf[off:off + 1] == b"\r":
        off += 1
    if buf[off:off + 1] == b"\n":
        off += 1
    fmt, els, cur = "ascii", [], None
    for line in buf[:end].decode("latin1").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            cur = {"name": t[1], "count": int(t[2]), "props": []}
            els.append(cur)
        elif t[0] == "property" and cur is not None:
            if t[1] == "list":
                cur["props"].append(("list", t[2], t[3], t[4]))
            else:
                cur["props"].append((t[1], t[2]))
    tmap = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
            "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
            "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
    vel = next((e for e in els if e["name"] == "vertex"), None)
    if vel is None:
        raise ValueError("PLY file has no vertex element.")
    names = [p[-1] for p in vel["props"]]
    if fmt == "ascii":
        lines = buf[off:].decode("latin1").splitlines()
        skip = 0
        for e in els:
            if e is vel:
                break
            skip += e["count"]
        rows = np.array([l.split()[:len(names)] for l in lines[skip:skip + vel["count"]]], dtype=np.float64)
        return rows[:, [names.index("x"), names.index("y"), names.index("z")]]
    end_ch = "<" if fmt == "binary_little_endian" else ">"
    pos = off
    for e in els:
        if e is vel:
            if any(p[0] == "list" for p in e["props"]):
                raise ValueError("Vertex element with list properties is not supported.")
            dt = np.dtype([(p[1], end_ch + tmap[p[0]]) for p in e["props"]])
            arr = np.frombuffer(buf, dtype=dt, count=e["count"], offset=pos)
            return np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
        # skip a preceding element
        for _ in range(e["count"]):
            for p in e["props"]:
                if p[0] == "list":
                    n = np.frombuffer(buf, end_ch + tmap[p[1]], 1, pos)[0]
                    pos += np.dtype(tmap[p[1]]).itemsize + int(n) * np.dtype(tmap[p[2]]).itemsize
                else:
                    pos += np.dtype(tmap[p[0]]).itemsize
    raise ValueError("Could not read vertices.")


def load_points(path: str, data: bytes | None = None) -> np.ndarray:
    """Load a point cloud from a path, or from bytes with `path` giving the file name."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        return load_ply(data if data is not None else path)
    pts = []
    text = data.decode("utf-8", "ignore") if data is not None else open(path, "r", errors="ignore").read()
    for line in text.splitlines():
        if ext == ".obj":
            if line.startswith("v "):
                t = line.split()
                pts.append((float(t[1]), float(t[2]), float(t[3])))
        else:
            t = line.replace(",", " ").replace(";", " ").split()
            if len(t) >= 3:
                try:
                    pts.append((float(t[0]), float(t[1]), float(t[2])))
                except ValueError:
                    pass
    if not pts:
        raise ValueError("No points found in " + path)
    return np.asarray(pts, np.float64)


def make_sample(seed: int = 42) -> np.ndarray:
    """Synthetic tabletop matching the web app's sample: notebook, phone, cable,
    headphones, open laptop and a bottle on an 80 x 60 cm table, posed as if
    captured by a handheld phone."""
    rng = np.random.default_rng(seed)
    W, D, st = 0.8, 0.6, 0.0028
    X, Y = np.meshgrid(np.arange(0, W + 1e-9, st), np.arange(0, D + 1e-9, st), indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    keep = rng.random(X.size) >= 0.02
    X = X[keep] + (rng.random(keep.sum()) - 0.5) * st * 0.8
    Y = Y[keep] + (rng.random(keep.sum()) - 0.5) * st * 0.8
    m = (X >= 0) & (Y >= 0) & (X <= W) & (Y <= D)
    X, Y = X[m], Y[m]

    def rect(cx, cy, w, d, a):
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        dx, dy = X - cx, Y - cy
        return (np.abs(c * dx + s * dy) <= w / 2) & (np.abs(-s * dx + c * dy) <= d / 2)

    def dpoly(pts):
        best = np.full(X.shape, 1e9)
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            vx, vy = bx - ax, by - ay
            t = np.clip(((X - ax) * vx + (Y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
            best = np.minimum(best, np.hypot(X - ax - t * vx, Y - ay - t * vy))
        return best

    Z = np.zeros_like(X)
    put = lambda mask, h: np.maximum(Z, np.where(mask, h, 0.0))
    Z = put(dpoly([(0.0, 0.07), (0.12, 0.05), (0.26, 0.09), (0.40, 0.06), (0.52, 0.16), (0.64, 0.22), (0.80, 0.20)]) < 0.0035, 0.006)
    Z = put(rect(0.20, 0.22, 0.21, 0.15, 12), 0.015)                       # notebook
    Z = put(rect(0.60, 0.09, 0.075, 0.155, -25), 0.009)                    # phone
    Z = put(dpoly([(0.40, 0.37), (0.45, 0.30), (0.44, 0.24), (0.39, 0.21)]) < 0.013, 0.018)  # headphone band
    Z = put(np.hypot(X - 0.40, Y - 0.37) < 0.042, 0.042)                   # ear cups
    Z = put(np.hypot(X - 0.39, Y - 0.21) < 0.042, 0.042)
    Z = put((X > 0.48) & (X < 0.76) & (Y > 0.27) & (Y < 0.46), 0.018)     # laptop base
    scr = (X > 0.48) & (X < 0.76) & (Y >= 0.46) & (Y < 0.532)
    Z = np.maximum(Z, np.where(scr, 0.018 + (Y - 0.46) * math.tan(math.radians(70)), 0))  # open screen
    Z = put(np.hypot(X - 0.28, Y - 0.50) < 0.035, 0.22)                    # bottle
    Z = Z + rng.normal(0, 0.0012, Z.size)

    ax, az = math.radians(35), math.radians(20)
    y1 = math.cos(ax) * Y - math.sin(ax) * Z
    z1 = math.sin(ax) * Y + math.cos(ax) * Z
    x2 = math.cos(az) * X - math.sin(az) * y1
    y2 = math.sin(az) * X + math.cos(az) * y1
    return np.stack([x2 + 0.1, y2 - 0.3, z1 - 0.45], 1)


# --------------------------------------------------------------------------- table frame
def fit_table(P: np.ndarray, units: str = "auto", flip: bool = False, seed: int = 7):
    """RANSAC the table plane, orient 'up' toward the objects, align axes with the
    table edges. Returns points as (u, v, h) in meters plus the table bounds."""
    rng = np.random.default_rng(seed)
    if units == "auto":
        s = P[:: max(1, len(P) // 20000)]
        diag = np.linalg.norm(s.max(0) - s.min(0))
        sc = 1.0 if diag < 20 else 0.01 if diag < 500 else 0.001
    else:
        sc = {"m": 1.0, "cm": 0.01, "mm": 0.001}[units]
    P = P * sc
    n = len(P)
    S = P[rng.integers(0, n, min(n, 5000))]
    thr, best, bc = 0.006, None, -1
    for _ in range(600):
        a, b, c = S[rng.integers(0, len(S), 3)]
        nn = np.cross(b - a, c - a)
        l = np.linalg.norm(nn)
        if l < 1e-10:
            continue
        nn /= l
        d = -nn @ a
        cnt = np.count_nonzero(np.abs(S @ nn + d) < thr)
        if cnt > bc:
            bc, best = cnt, (nn, d)
    if best is None:
        raise ValueError("Could not find a flat surface in this scan.")
    nv, d = best
    sub = P[:: max(1, n // 150000)]
    cen = sub.mean(0)
    for _ in range(2):
        inl = sub[np.abs(sub @ nv + d) < thr]
        if len(inl) < 10:
            break
        cen = inl.mean(0)
        w, V = np.linalg.eigh(np.cov((inl - cen).T))
        nn = V[:, 0]
        nv = nn if nn @ nv >= 0 else -nn
        d = -nv @ cen
    sd = S @ nv + d
    m = (np.abs(sd) > 0.015) & (np.abs(sd) < 0.4)
    if (sd[m] < 0).sum() > (sd[m] > 0).sum():
        nv = -nv
    if flip:
        nv = -nv
    e1 = np.cross(nv, [1, 0, 0] if abs(nv[0]) < 0.9 else [0, 1, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nv, e1)
    X = sub - cen
    inl = X[np.abs(X @ nv) < thr]
    inl = inl[:: max(1, len(inl) // 6000)]
    su, sv = inl @ e1, inl @ e2
    best_a, best_area, bw, bh = 0.0, np.inf, 0, 0
    for deg in np.arange(0, 90, 0.5):
        a = math.radians(deg)
        ru = math.cos(a) * su + math.sin(a) * sv
        rv = -math.sin(a) * su + math.cos(a) * sv
        w = np.subtract(*np.percentile(ru, [99.5, 0.5]))
        h = np.subtract(*np.percentile(rv, [99.5, 0.5]))
        if w * h < best_area:
            best_a, best_area, bw, bh = a, w * h, w, h
    if bh > bw:
        best_a += math.pi / 2
    e1 = math.cos(best_a) * e1 + math.sin(best_a) * e2
    e1 /= np.linalg.norm(e1)
    # deterministic sign so the web app and this script agree on which corner is (0, 0)
    ref = np.array([1.0, 0, 0]) if abs(e1[0]) > 0.2 else np.array([0, 1.0, 0])
    if e1 @ ref < 0:
        e1 = -e1
    e2 = np.cross(nv, e1)
    Xa = P - cen
    uvh = np.stack([Xa @ e1, Xa @ e2, Xa @ nv], 1)
    tab = uvh[:: max(1, n // 150000)]
    tab = tab[np.abs(tab[:, 2]) < thr]
    pad = 0.015
    u0, u1 = np.percentile(tab[:, 0], [0.3, 99.7])
    v0, v1 = np.percentile(tab[:, 1], [0.3, 99.7])
    bounds = (u0 - pad, u1 + pad, v0 - pad, v1 + pad)
    return uvh, bounds, bc / len(S)


# --------------------------------------------------------------------------- grid + classes
def build_grid(uvh, bounds, cell_cm=1.0):
    u0, u1, v0, v1 = bounds
    cs = cell_cm / 100
    gw, gh = int(math.ceil((u1 - u0) / cs)), int(math.ceil((v1 - v0) / cs))
    h = uvh[:, 2]
    ix = np.floor((uvh[:, 0] - u0) / cs).astype(int)
    iy = np.floor((uvh[:, 1] - v0) / cs).astype(int)
    ok = (h >= -0.015) & (h <= 1.5) & (ix >= 0) & (iy >= 0) & (ix < gw) & (iy < gh)
    k, h = iy[ok] * gw + ix[ok], h[ok]
    order = np.lexsort((h, k))
    ks, hs = k[order], h[order]
    last = np.r_[ks[1:] != ks[:-1], True]
    li = np.nonzero(last)[0]
    cnt = np.bincount(ks, minlength=gw * gh)
    H = np.full(gw * gh, np.nan)
    cells = ks[li]
    second = np.where(cnt[cells] >= 3, hs[np.maximum(li - 1, 0)], hs[li])
    H[cells] = np.maximum(second, 0)
    H = H.reshape(gh, gw)
    K = np.ones((3, 3)); K[1, 1] = 0
    for _ in range(3):
        known = ~np.isnan(H)
        s = ndimage.convolve(np.where(known, H, 0), K, mode="constant")
        c = ndimage.convolve(known.astype(float), K, mode="constant")
        fill = ~known & (c >= 5)
        H[fill] = s[fill] / c[fill]
    # 3x3 median over known cells removes single-cell LiDAR spikes ("flying pixels")
    known = ~np.isnan(H)
    pad = np.pad(H, 1, constant_values=np.nan)
    stack = np.stack([pad[1 + dy:1 + dy + gh, 1 + dx:1 + dx + gw] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
    H = np.where(known, med, np.nan)
    return H, cs


def classify(H, cs, step_cm=2.5, tilt_deg=25, radius_cm=6, penalty=4.0):
    step = step_cm / 100
    known = ~np.isnan(H)
    tall = known & (H > step)
    T = np.where(known & ~tall, H, np.nan)
    k = max(1, round(radius_cm / 100 / cs))
    hi = ndimage.maximum_filter(np.where(np.isnan(T), -np.inf, T), 2 * k + 1, mode="constant", cval=-np.inf)
    lo = ndimage.minimum_filter(np.where(np.isnan(T), np.inf, T), 2 * k + 1, mode="constant", cval=np.inf)
    with np.errstate(invalid="ignore"):
        tilt = np.where(hi > lo, np.degrees(np.arctan((hi - lo) / (2 * k * cs))), 0.0)
    m = ~np.isnan(T)
    T0 = np.where(m, T, 0)
    one = np.ones((3, 3))
    c = ndimage.convolve(m.astype(float), one, mode="constant")
    s1 = ndimage.convolve(T0, one, mode="constant")
    s2 = ndimage.convolve(T0 * T0, one, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        rough = np.sqrt(np.maximum(0, s2 / c - (s1 / c) ** 2))
    rough = np.nan_to_num(rough)

    cls = np.full(H.shape, FREE, np.uint8)
    cls[(np.nan_to_num(H) > FLAT) | (rough > ROUGH)] = CLIMB
    cls[tilt > tilt_deg] = OBST
    cls[tall] = OBST
    cls[~known] = VOID
    r = radius_cm / 100 / cs
    R = int(math.ceil(r))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    disk = xx * xx + yy * yy <= r * r
    blocked = np.pad((cls == VOID) | (cls == OBST), 1, constant_values=True)
    grown = ndimage.binary_dilation(blocked, structure=disk)[1:-1, 1:-1]
    cls[grown & ((cls == FREE) | (cls == CLIMB))] = MARGIN

    tp = 0.5 * np.minimum(1, tilt / tilt_deg)
    cost = np.full(H.shape, np.inf)
    cost[cls == FREE] = 1 + tp[cls == FREE]
    cm_ = cls == CLIMB
    cost[cm_] = 1 + penalty * np.minimum(1, H[cm_] / step) + tp[cm_]
    return cls, cost, tilt


# --------------------------------------------------------------------------- planning
def snap(cost, x, y):
    gh, gw = cost.shape
    x, y = int(np.clip(x, 0, gw - 1)), int(np.clip(y, 0, gh - 1))
    if np.isfinite(cost[y, x]):
        return x, y, 0.0
    ys, xs = np.nonzero(np.isfinite(cost))
    if len(xs) == 0:
        return None
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    i = int(np.argmin(d2))
    return int(xs[i]), int(ys[i]), math.sqrt(d2[i])


def astar(cost, start, goal):
    gh, gw = cost.shape
    c = cost.ravel()
    s, t = start[1] * gw + start[0], goal[1] * gw + goal[0]
    tx, ty = goal
    g = np.full(gw * gh, np.inf)
    came = np.full(gw * gh, -1, np.int64)
    closed = np.zeros(gw * gh, bool)
    SQ = math.sqrt(2)

    def h(i):
        dx, dy = abs(i % gw - tx), abs(i // gw - ty)
        return max(dx, dy) + (SQ - 1) * min(dx, dy)

    g[s] = 0
    pq = [(h(s), s)]
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while pq:
        _, cur = heapq.heappop(pq)
        if closed[cur]:
            continue
        closed[cur] = True
        if cur == t:
            break
        cx, cy = cur % gw, cur // gw
        for dx, dy in nbrs:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= gw or ny >= gh:
                continue
            n = ny * gw + nx
            if closed[n] or not math.isfinite(c[n]):
                continue
            diag = dx != 0 and dy != 0
            if diag and (not math.isfinite(c[cy * gw + nx]) or not math.isfinite(c[ny * gw + cx])):
                continue
            ng = g[cur] + (SQ if diag else 1) * (c[cur] + c[n]) / 2
            if ng < g[n]:
                g[n], came[n] = ng, cur
                heapq.heappush(pq, (ng + h(n), n))
    if not closed[t]:
        return None
    path, cur = [], t
    while cur != -1:
        path.append((cur % gw, cur // gw))
        cur = came[cur]
    return path[::-1]


# --------------------------------------------------------------------------- MuJoCo world
def hazard_texture(cls, scale=6):
    """Class colours as a PNG, upsampled so cells stay crisp. Row 0 of the image is
    the far (+y) edge of the table, which is how MuJoCo maps 2D textures onto hfields."""
    import imageio.v3 as iio
    img = CLASS_RGB[cls][::-1]
    img = np.repeat(np.repeat(img, scale, 0), scale, 1)
    return iio.imwrite("<bytes>", img, extension=".png")


def build_xml(H, cls, cs, path_xy, rover):
    gh, gw = H.shape
    elev = np.where(np.isnan(H), 0.0, TABLE_Z + np.nan_to_num(H))
    zmax = float(elev.max())
    norm_ = elev / zmax
    sx, sy = (gw - 1) * cs / 2, (gh - 1) * cs / 2
    cx, cy = 0.5 * cs + sx, 0.5 * cs + sy
    data = " ".join(f"{v:.5f}" for v in norm_[::-1].ravel())   # MuJoCo hfield row 0 is the +y edge
    rw, a, b = rover["wheel"], rover["half_base"], rover["half_track"]
    ch_l, ch_w = a + rw * 0.3, b - 0.009
    sites = []
    for i, (x, y, z) in enumerate(path_xy[::2]):
        sites.append(f'<site type="sphere" size="0.0028" pos="{x:.4f} {y:.4f} {z + 0.003:.4f}" rgba="0.14 0.26 0.77 0.95"/>')
    (ax_, ay_, az_), (bx_, by_, bz_) = path_xy[0], path_xy[-1]
    span = max(gw, gh) * cs
    wheels = []
    for nm, fx, fy in [("fl", a, b), ("fr", a, -b), ("rl", -a, b), ("rr", -a, -b)]:
        wheels.append(f'''
      <body name="wheel_{nm}" pos="{fx:.4f} {fy:.4f} 0">
        <joint name="j_{nm}" type="hinge" axis="0 1 0" damping="0.0004"/>
        <geom type="sphere" size="{rw:.4f}" mass="0.03" group="3" rgba="0 0 0 0" friction="1.6 0.02 0.002" contype="2" conaffinity="1"/>
        <geom type="cylinder" size="{rw:.4f} 0.006" euler="1.5708 0 0" mass="0.001" rgba="0.13 0.14 0.16 1" contype="0" conaffinity="0"/>
        <geom type="cylinder" size="{rw * 0.55:.4f} 0.0065" euler="1.5708 0 0" mass="0.001" rgba="0.75 0.77 0.8 1" contype="0" conaffinity="0"/>
      </body>''')
    acts = "\n    ".join(f'<velocity name="v_{nm}" joint="j_{nm}" kv="{rover["kv"]}" forcerange="-{rover["torque"]} {rover["torque"]}" ctrlrange="-60 60"/>'
                           for nm in ["fl", "fr", "rl", "rr"])
    return f'''
<mujoco model="tabletop_traverse">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <map znear="0.002" zfar="20"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.16 0.19 0.24" rgb2="0.03 0.04 0.06" width="256" height="256"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.2 0.22 0.25" rgb2="0.17 0.19 0.22" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="10 10" reflectance="0.05"/>
    <texture name="haz" type="2d" file="hazard.png"/>
    <material name="haz" texture="haz" specular="0.15" shininess="0.2"/>
    <hfield name="terrain" nrow="{gh}" ncol="{gw}" size="{sx:.5f} {sy:.5f} {zmax:.5f} 0.02" elevation="{data}"/>
  </asset>
  <worldbody>
    <light pos="{cx:.3f} {cy - 0.4:.3f} 1.2" dir="0 0.35 -1" diffuse="0.65 0.65 0.65" castshadow="true"/>
    <geom name="floor" type="plane" size="{span * 2:.3f} {span * 2:.3f} 0.1" pos="{cx:.4f} {cy:.4f} -0.0015" material="floor" contype="1" conaffinity="2"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="{cx:.5f} {cy:.5f} 0" material="haz" contype="1" conaffinity="2" friction="1.2 0.02 0.002"/>
    {"".join(sites)}
    <site name="A" type="cylinder" size="0.008 0.0015" pos="{ax_:.4f} {ay_:.4f} {az_ + 0.0015:.4f}" rgba="0.14 0.26 0.77 1"/>
    <site name="B" type="cylinder" size="0.012 0.0015" pos="{bx_:.4f} {by_:.4f} {bz_ + 0.0015:.4f}" rgba="0.2 0.75 0.45 1"/>
    <site name="B_flag" type="capsule" size="0.0015 0.02" pos="{bx_:.4f} {by_:.4f} {bz_ + 0.02:.4f}" rgba="0.2 0.75 0.45 1"/>
    <body name="rover" pos="{ax_:.4f} {ay_:.4f} {az_ + rw + 0.004:.4f}">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size="{ch_l:.4f} {ch_w:.4f} 0.006" mass="0.3" rgba="0.93 0.94 0.96 1" contype="2" conaffinity="1"/>
      <geom type="box" size="{ch_l * 0.5:.4f} {ch_w * 0.8:.4f} 0.0015" pos="{-ch_l * 0.25:.4f} 0 0.0072" mass="0.005" rgba="0.12 0.2 0.45 1" contype="0" conaffinity="0"/>
      <geom type="capsule" size="0.0025 0.012" pos="{ch_l * 0.6:.4f} 0 0.018" mass="0.003" rgba="0.8 0.82 0.85 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.004 0.009 0.004" pos="{ch_l * 0.6 + 0.002:.4f} 0 0.032" mass="0.003" rgba="0.14 0.26 0.77 1" contype="0" conaffinity="0"/>
      <site name="imu"/>
      {"".join(wheels)}
    </body>
  </worldbody>
  <actuator>
    {acts}
  </actuator>
</mujoco>'''


# --------------------------------------------------------------------------- controller
class PurePursuit:
    """Skid-steer pure pursuit along the planned waypoints."""

    def __init__(self, path_xy, rover, lookahead=0.06, vmax=0.08):
        self.P = np.asarray(path_xy)[:, :2]
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.s = np.r_[0, np.cumsum(seg)]
        self.idx = 0
        self.L = lookahead
        self.vmax = vmax
        self.rw = rover["wheel"]
        self.tw = 2 * rover["half_track"]

    def __call__(self, x, y, yaw):
        lo, hi = self.idx, min(len(self.P), self.idx + 40)
        d = np.hypot(self.P[lo:hi, 0] - x, self.P[lo:hi, 1] - y)
        self.idx = lo + int(np.argmin(d))
        cte = float(d.min())
        target_s = self.s[self.idx] + self.L
        j = min(int(np.searchsorted(self.s, target_s)), len(self.P) - 1)
        tx, ty = self.P[j]
        alpha = (math.atan2(ty - y, tx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
        dist_goal = math.hypot(self.P[-1, 0] - x, self.P[-1, 1] - y)
        if abs(alpha) > math.radians(55):
            v = 0.0                                   # turn in place first
        else:
            v = self.vmax * math.cos(alpha) * min(1.0, dist_goal / 0.05 + 0.3)
        w = float(np.clip(3.5 * alpha, -2.5, 2.5))
        wl = (v - w * self.tw / 2) / self.rw
        wr = (v + w * self.tw / 2) / self.rw
        return wl, wr, cte, dist_goal


def yaw_tilt(data, bid):
    R = data.xmat[bid].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
    return yaw, tilt




# --------------------------------------------------------------------------- orchestration (shared by CLI and server)
def rover_params(radius_cm: float, wheel: float = 0.022) -> dict:
    """Rover geometry sized so its footprint roughly matches the planner's radius."""
    half = max(wheel + 0.005, (radius_cm / 100) / math.sqrt(2) - wheel)
    return {"wheel": wheel, "half_base": half, "half_track": half + 0.008, "kv": 0.05, "torque": 0.12}


def elevation(H: np.ndarray) -> np.ndarray:
    """Sim elevation: table top raised by TABLE_Z, off-surface cells at floor level."""
    return np.where(np.isnan(H), 0.0, TABLE_Z + np.nan_to_num(H))


def auto_ab(cost: np.ndarray):
    """Two cells far apart (by driving distance) in the largest drivable region."""
    from collections import deque
    ok = np.isfinite(cost)
    lab, n = ndimage.label(ok)            # 4-connected: A* never squeezes diagonally between blocked cells
    if n == 0:
        return None
    big = int(np.argmax(np.bincount(lab.ravel())[1:])) + 1
    M = lab == big
    gh, gw = M.shape

    def far(s):
        seen = np.zeros_like(M)
        seen[s[1], s[0]] = True
        q, last = deque([s]), s
        while q:
            x, y = q.popleft()
            last = (x, y)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < gw and 0 <= ny < gh and M[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((nx, ny))
        return last

    ys, xs = np.nonzero(M)
    a = far((int(xs[0]), int(ys[0])))
    b = far(a)
    if a[0] > b[0]:
        a, b = b, a
    return a, b


def plan_route(H, cs, cost, a_cm=None, b_cm=None) -> dict:
    """Snap A/B to safe cells and run A*. A/B are cm from the map's lower-left corner;
    if either is missing, a far-apart pair in the main drivable region is chosen."""
    gh, gw = cost.shape
    if a_cm is None or b_cm is None:
        ab = auto_ab(cost)
        if ab is None:
            return {"error": "There is no drivable surface on this map."}
        ca, cb = ab
    else:
        ca = (int(a_cm[0] / 100 / cs), int(a_cm[1] / 100 / cs))
        cb = (int(b_cm[0] / 100 / cs), int(b_cm[1] / 100 / cs))
    A, B = snap(cost, *ca), snap(cost, *cb)
    if A is None or B is None:
        return {"error": "There is no drivable surface on this map."}
    out = {"a_cell": [A[0], A[1]], "b_cell": [B[0], B[1]],
           "a_moved_cm": round(A[2] * cs * 100, 1), "b_moved_cm": round(B[2] * cs * 100, 1)}
    path = astar(cost, A[:2], B[:2])
    if path is None:
        out["error"] = "No safe route: obstacles cut A off from B."
        return out
    elev = elevation(H)
    path_xy = [((x + 0.5) * cs, (y + 0.5) * cs, float(elev[y, x])) for x, y in path]
    seg = np.linalg.norm(np.diff(np.asarray(path_xy)[:, :2], axis=0), axis=1)
    out.update(path=[[int(x), int(y)] for x, y in path], path_xy=path_xy, length=float(seg.sum()))
    return out


class RoverRun:
    """One MuJoCo rover drive along a planned route. Step it, read poses, get a summary."""

    CTRL_EVERY = 10  # 50 Hz control at 2 ms physics

    def __init__(self, H, cls, cs, path_xy, rover, speed=0.08, seconds=60.0):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(build_xml(H, cls, cs, path_xy, rover),
                                                    assets={"hazard.png": hazard_texture(cls)})
        self.data = mujoco.MjData(self.model)
        self.bid = self.model.body("rover").id
        self.wheel_adr = [int(self.model.jnt_qposadr[self.model.joint(f"j_{n}").id]) for n in ("fl", "fr", "rl", "rr")]
        j = min(8, len(path_xy) - 1)
        self.yaw0 = math.atan2(path_xy[j][1] - path_xy[0][1], path_xy[j][0] - path_xy[0][0])
        self.data.qpos[3:7] = [math.cos(self.yaw0 / 2), 0, 0, math.sin(self.yaw0 / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.path_xy, self.rover, self.seconds = path_xy, rover, seconds
        self.ctrl = PurePursuit(path_xy, rover, vmax=speed)
        seg = np.linalg.norm(np.diff(np.asarray(path_xy)[:, :2], axis=0), axis=1)
        self.planned = float(seg.sum())
        self.status, self.done, self.k = "driving", False, 0
        self.max_tilt = self.max_cte = self.driven = 0.0
        self.last_xy, self.best_idx, self.best_t = None, 0, 0.0
        self.log = {"t": [], "x": [], "y": [], "tilt": [], "cte": []}

    def control_step(self):
        d = self.data
        x, y = d.xpos[self.bid][:2]
        yaw, tl = yaw_tilt(d, self.bid)
        wl, wr, cte, dgoal = self.ctrl(x, y, yaw)
        if d.time < 0.4:                      # let the rover settle on its wheels
            wl = wr = 0.0
        d.ctrl[:] = [wl, wr, wl, wr]
        if self.last_xy is not None:
            self.driven += math.hypot(x - self.last_xy[0], y - self.last_xy[1])
        self.last_xy = (x, y)
        self.max_tilt, self.max_cte = max(self.max_tilt, tl), max(self.max_cte, cte)
        L = self.log
        L["t"].append(round(d.time, 3)); L["x"].append(round(float(x), 4)); L["y"].append(round(float(y), 4))
        L["tilt"].append(round(tl, 2)); L["cte"].append(round(cte, 4))
        if self.ctrl.idx > self.best_idx:
            self.best_idx, self.best_t = self.ctrl.idx, d.time
        if dgoal < 0.02:
            self.status = "reached goal"
        elif tl > 65:
            self.status = "tipped over"
        elif d.xpos[self.bid][2] < TABLE_Z * 0.5:
            self.status = "fell off the table"
        elif d.time - self.best_t > 6 and d.time > 2:
            self.status = "stuck"
        if self.status != "driving":
            self.done = True
            d.ctrl[:] = 0

    def step(self, n=1):
        for _ in range(n):
            if not self.done and self.k % self.CTRL_EVERY == 0:
                self.control_step()
            self.mj.mj_step(self.model, self.data)
            self.k += 1
            if not self.done and self.data.time >= self.seconds:
                self.status, self.done = "timeout", True
                self.data.ctrl[:] = 0

    def pose(self):
        """[t, x, y, z, qw, qx, qy, qz, wheel_fl, wheel_fr, wheel_rl, wheel_rr]"""
        d = self.data
        p, q = d.xpos[self.bid], d.xquat[self.bid]
        return [round(float(d.time), 3)] + [round(float(v), 5) for v in p] + [round(float(v), 5) for v in q] + \
               [round(float(d.qpos[a]), 3) for a in self.wheel_adr]

    def summary(self) -> dict:
        return {"status": self.status, "sim_time_s": round(float(self.data.time), 2),
                "planned_cm": round(self.planned * 100, 1), "driven_cm": round(self.driven * 100, 1),
                "max_tilt_deg": round(self.max_tilt, 1), "max_tracking_error_cm": round(self.max_cte * 100, 2)}


def record_frames(run: RoverRun, fps: int = 30, hold_s: float = 1.0):
    """Run to completion, sampling poses at `fps`; keep `hold_s` of frames after the end."""
    steps = max(1, int(round(1 / fps / run.model.opt.timestep)))
    frames, hold = [run.pose()], 0
    while hold < fps * hold_s:
        run.step(steps)
        frames.append(run.pose())
        if run.done:
            hold += 1
    return frames
