"""
Tabletop Traverse - web server

  uvicorn server:app --port 7860            # then open http://localhost:7860

POST /api/scan       upload a point cloud (or pick a built-in one) -> scan_id
POST /api/map        hazard map + A* route for a scan and rover settings
POST /api/simulate   headless MuJoCo drive along the route -> terrain, route and rover poses
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rover_core as R

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_UPLOAD = 80 * 1024 * 1024
MAX_POINTS = 1_500_000
BUILTINS = {
    "our_table": ("Our table (iPhone LiDAR, Scaniverse)", os.path.join(HERE, "scans", "our_table.ply")),
    "sample": ("Synthetic sample table", None),
}

app = FastAPI(title="Tabletop Traverse")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

_scans: "OrderedDict[str, dict]" = OrderedDict()
_lock = threading.Lock()
_heavy = threading.Semaphore(2)          # at most two fits / simulations at once


def _remember(scan_id: str, entry: dict):
    with _lock:
        _scans[scan_id] = entry
        _scans.move_to_end(scan_id)
        while len(_scans) > 16:
            _scans.popitem(last=False)


def _get(scan_id: str) -> dict:
    with _lock:
        e = _scans.get(scan_id)
        if e is not None:
            _scans.move_to_end(scan_id)
    if e is None:
        # built-ins can always be rebuilt (e.g. after a server restart)
        for bid in BUILTINS:
            if scan_id.startswith(f"builtin:{bid}:"):
                _, _, units, flip = scan_id.split(":")
                return _load_builtin(bid, units, flip == "1")
        raise HTTPException(404, "That scan has expired. Load it again.")
    return e


def _fit(raw: np.ndarray, name: str, units: str, flip: bool, scan_id: str) -> dict:
    if len(raw) > MAX_POINTS:
        raw = raw[np.random.default_rng(0).choice(len(raw), MAX_POINTS, replace=False)]
    if len(raw) < 300:
        raise HTTPException(400, "That scan has too few points to map.")
    with _heavy:
        try:
            uvh, bounds, inl = R.fit_table(raw, units, flip)
        except ValueError as e:
            raise HTTPException(400, str(e))
    u0, u1, v0, v1 = bounds
    if (u1 - u0) * (v1 - v0) > 9.0:
        raise HTTPException(400, "The flat surface in this scan is larger than 3 x 3 m. Crop the scan to the tabletop.")
    entry = {"scan_id": scan_id, "name": name, "points": int(len(raw)), "plane_fit": round(float(inl), 2),
             "size_cm": [round((u1 - u0) * 100, 1), round((v1 - v0) * 100, 1)], "uvh": uvh, "bounds": bounds,
             "grids": {}}
    _remember(scan_id, entry)
    return entry


def _load_builtin(bid: str, units: str = "auto", flip: bool = False) -> dict:
    name, path = BUILTINS[bid]
    raw = R.make_sample() if path is None else R.load_points(path)
    return _fit(raw, name, units, flip, f"builtin:{bid}:{units}:{int(flip)}")


def _public(e: dict) -> dict:
    return {k: e[k] for k in ("scan_id", "name", "points", "plane_fit", "size_cm")}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/scan")
async def scan(file: UploadFile | None = File(None), builtin: str | None = Form(None),
               units: str = Form("auto"), flip: bool = Form(False)):
    if units not in ("auto", "m", "cm", "mm"):
        raise HTTPException(400, "Units must be auto, m, cm or mm.")
    if file is None:
        bid = builtin or "our_table"
        if bid not in BUILTINS:
            raise HTTPException(400, "Unknown built-in scan.")
        from starlette.concurrency import run_in_threadpool
        return _public(await run_in_threadpool(_load_builtin, bid, units, flip))
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "That file is over 80 MB. Export a lower-density point cloud.")
    fname = os.path.basename(file.filename or "scan.ply")
    if os.path.splitext(fname)[1].lower() not in (".ply", ".obj", ".xyz", ".txt", ".csv", ".pts"):
        raise HTTPException(400, "Upload a PLY point cloud (OBJ and XYZ also work).")
    scan_id = hashlib.sha1(data).hexdigest()[:16] + f":{units}:{int(flip)}"
    with _lock:
        if scan_id in _scans:
            return _public(_scans[scan_id])

    def work():
        try:
            raw = R.load_points(fname, data)
        except Exception as e:  # noqa: BLE001 - report any parse failure to the user
            raise HTTPException(400, f"Could not read {fname}: {e}")
        return _fit(raw, fname, units, flip, scan_id)

    from starlette.concurrency import run_in_threadpool
    return _public(await run_in_threadpool(work))


class Params(BaseModel):
    step: float = Field(2.5, ge=0.3, le=10)
    tilt: float = Field(25, ge=5, le=60)
    radius: float = Field(6, ge=1, le=15)
    penalty: float = Field(4, ge=0, le=20)
    cell: float = Field(1.0, ge=0.5, le=3)


class MapReq(BaseModel):
    scan_id: str
    params: Params = Params()
    a: list[float] | None = None
    b: list[float] | None = None


class SimReq(MapReq):
    speed: float = Field(8, ge=2, le=20)


def _map(req: MapReq):
    e = _get(req.scan_id)
    p = req.params
    key = round(p.cell, 3)
    if key not in e["grids"]:
        e["grids"][key] = R.build_grid(e["uvh"], e["bounds"], p.cell)
    H, cs = e["grids"][key]
    cls, cost, _ = R.classify(H, cs, p.step, p.tilt, p.radius, p.penalty)
    a = req.a if req.a and len(req.a) == 2 else None
    b = req.b if req.b and len(req.b) == 2 else None
    plan = R.plan_route(H, cs, cost, a, b)
    return e, H, cs, cls, plan


def _map_json(e, H, cs, cls, plan) -> dict:
    gh, gw = H.shape
    heights = np.where(np.isnan(H), -1, np.round(np.nan_to_num(H) * 1000, 1))   # mm, -1 = no data
    out = {"scan": _public(e), "gw": gw, "gh": gh, "cell_cm": round(cs * 100, 4),
           "heights_mm": heights.ravel().tolist(), "classes": cls.ravel().tolist(),
           "counts": np.bincount(cls.ravel(), minlength=5).tolist()}
    pj = {k: plan[k] for k in ("a_moved_cm", "b_moved_cm", "error") if k in plan}
    if "a_cell" in plan:
        pj["a_cm"] = [round((plan["a_cell"][0] + 0.5) * cs * 100, 2), round((plan["a_cell"][1] + 0.5) * cs * 100, 2)]
        pj["b_cm"] = [round((plan["b_cell"][0] + 0.5) * cs * 100, 2), round((plan["b_cell"][1] + 0.5) * cs * 100, 2)]
    if "path" in plan:
        pj["path"] = plan["path"]
        pj["length_cm"] = round(plan["length"] * 100, 1)
        pj["climb_cm"] = round(sum(
            np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(plan["path"][:-1], plan["path"][1:])
            if cls[y0, x0] == R.CLIMB or cls[y1, x1] == R.CLIMB) * cs * 100, 1)
        pj["max_height_cm"] = round(max(float(np.nan_to_num(H[y, x])) for x, y in plan["path"]) * 100, 2)
    out["plan"] = pj
    return out


@app.post("/api/map")
def map_(req: MapReq):
    return _map_json(*_map(req))


@app.post("/api/simulate")
def simulate(req: SimReq):
    e, H, cs, cls, plan = _map(req)
    out = _map_json(e, H, cs, cls, plan)
    if "path_xy" not in plan:
        raise HTTPException(400, plan.get("error", "No route to simulate."))
    rover = R.rover_params(req.params.radius)
    with _heavy:
        run = R.RoverRun(H, cls, cs, plan["path_xy"], rover, speed=req.speed / 100, seconds=90)
        frames = R.record_frames(run, fps=30, hold_s=1.0)
    out.update(
        table_z=R.TABLE_Z,
        elevation_mm=np.round(R.elevation(H) * 1000, 1).ravel().tolist(),
        route_xyz=[[round(x, 4), round(y, 4), round(z, 4)] for x, y, z in plan["path_xy"]],
        rover={"wheel": rover["wheel"], "half_base": rover["half_base"], "half_track": rover["half_track"]},
        fps=30, frames=frames, result=run.summary())
    return out
