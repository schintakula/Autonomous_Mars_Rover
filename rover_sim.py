#!/usr/bin/env python3
"""
Tabletop Traverse - MuJoCo rover simulation (command line)

  python rover_sim.py                                     # sample table -> rover_run.mp4
  python rover_sim.py --scan scans/our_table.ply --penalty 0
  python rover_sim.py --scan table.ply --a 24.5 38.5 --b 117.5 87.5
  mjpython rover_sim.py --view                            # live viewer (macOS needs mjpython)

A / B are cm from the table's lower-left corner (the numbers the web app shows on hover).
Omit them to auto-pick two far-apart points in the main drivable region.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

import rover_core as R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", help="PLY / OBJ / XYZ point cloud. Omit to use the synthetic sample table.")
    ap.add_argument("--units", default="auto", choices=["auto", "m", "cm", "mm"])
    ap.add_argument("--flip", action="store_true", help="Flip 'up' if objects come out as pits")
    ap.add_argument("--a", nargs=2, type=float, metavar=("X", "Y"), help="Start, cm from the lower-left corner")
    ap.add_argument("--b", nargs=2, type=float, metavar=("X", "Y"), help="Goal, cm from the lower-left corner")
    ap.add_argument("--step", type=float, default=2.5, help="Max climbable step, cm")
    ap.add_argument("--tilt", type=float, default=25, help="Max rover tilt, degrees")
    ap.add_argument("--radius", type=float, default=6, help="Rover radius for obstacle inflation, cm")
    ap.add_argument("--penalty", type=float, default=4, help="Cost for driving over small objects (0 = drive over)")
    ap.add_argument("--cell", type=float, default=1.0, help="Map cell size, cm")
    ap.add_argument("--speed", type=float, default=8, help="Top speed, cm/s")
    ap.add_argument("--seconds", type=float, default=90, help="Time limit, s")
    ap.add_argument("--out", default="rover_run.mp4", help="Video file to write")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--view", action="store_true", help="Interactive MuJoCo viewer instead of a video")
    ap.add_argument("--no-video", action="store_true", help="Physics only; write the JSON log")
    args = ap.parse_args()

    if not args.view and not args.no_video:
        os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
    import mujoco

    t0 = time.time()
    raw = R.load_points(args.scan) if args.scan else R.make_sample()
    name = os.path.basename(args.scan) if args.scan else "sample table"
    uvh, bounds, inl = R.fit_table(raw, args.units, args.flip)
    H, cs = R.build_grid(uvh, bounds, args.cell)
    cls, cost, _ = R.classify(H, cs, args.step, args.tilt, args.radius, args.penalty)
    gh, gw = H.shape
    print(f"[scan]  {name}: {len(raw):,} points, plane fit {inl:.0%}, map {gw}x{gh} cells at {cs * 100:g} cm")

    plan = R.plan_route(H, cs, cost, args.a, args.b)
    if "error" in plan:
        sys.exit("[plan]  " + plan["error"] + " Try a smaller --radius or larger --step.")
    for lbl in ("a", "b"):
        if plan[f"{lbl}_moved_cm"] > 0:
            print(f"[plan]  {lbl.upper()} moved {plan[f'{lbl}_moved_cm']} cm to the nearest safe cell")
    pa, pb = plan["path_xy"][0], plan["path_xy"][-1]
    print(f"[plan]  A ({pa[0] * 100:.1f}, {pa[1] * 100:.1f}) -> B ({pb[0] * 100:.1f}, {pb[1] * 100:.1f}) cm, "
          f"route {plan['length'] * 100:.1f} cm")

    rover = R.rover_params(args.radius)
    run = R.RoverRun(H, cls, cs, plan["path_xy"], rover, speed=args.speed / 100, seconds=args.seconds)
    model, data, bid = run.model, run.data, run.bid

    if args.view:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.lookat[:] = [gw * cs / 2, gh * cs / 2, R.TABLE_Z]
            v.cam.distance, v.cam.elevation, v.cam.azimuth = max(gw, gh) * cs * 1.3, -50, 90
            while v.is_running():
                tick = time.time()
                run.step(run.CTRL_EVERY)
                v.sync()
                time.sleep(max(0, run.CTRL_EVERY * model.opt.timestep - (time.time() - tick)))
    elif args.no_video:
        while not run.done:
            run.step(run.CTRL_EVERY)
    else:
        import imageio.v2 as imageio
        W, Hh, rw = 640, 480, rover["wheel"]
        renderer = mujoco.Renderer(model, Hh, W)
        over = mujoco.MjvCamera()
        over.lookat[:] = [gw * cs / 2, gh * cs / 2, R.TABLE_Z]
        over.distance, over.elevation, over.azimuth = max(gw, gh) * cs * 1.25, -52, 90
        chase = mujoco.MjvCamera()
        chase.distance, chase.elevation = 0.42, -24
        chase_az = math.degrees(run.yaw0)
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            Image = None
        trail = []
        writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
        steps = int(round(1 / args.fps / model.opt.timestep))
        hold = 0
        while hold < args.fps:
            run.step(steps)
            if run.done:
                hold += 1
            pos = data.xpos[bid].copy()
            if not trail or np.linalg.norm(pos[:2] - trail[-1][:2]) > 0.004:
                trail.append(pos)
            yaw, tl = R.yaw_tilt(data, bid)
            chase_az += 0.08 * ((math.degrees(yaw) - chase_az + 180) % 360 - 180)
            chase.lookat[:], chase.azimuth = pos, chase_az
            panels = []
            for cam in (over, chase):
                renderer.update_scene(data, camera=cam)
                scn = renderer.scene
                for p in trail[-400:]:
                    if scn.ngeom >= scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                        np.array([0.0022, 0, 0]), np.array([p[0], p[1], p[2] - rw + 0.003]),
                                        np.eye(3).ravel(), np.array([1.0, 0.55, 0.1, 1.0], np.float32))
                    scn.ngeom += 1
                panels.append(renderer.render())
            frame = np.concatenate(panels, axis=1)
            if Image is not None:
                im = Image.fromarray(frame)
                dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, frame.shape[1], 24], fill=(10, 14, 20))
                dr.text((10, 6), f"t {data.time:5.1f} s   driven {run.driven * 100:5.1f} cm / planned "
                                 f"{run.planned * 100:.1f} cm   tilt {tl:4.1f} deg   {run.status}", fill=(226, 232, 238))
                dr.rectangle([0, frame.shape[0] - 22, frame.shape[1], frame.shape[0]], fill=(10, 14, 20))
                dr.text((10, frame.shape[0] - 16), "Overview: blue = planned route, orange = driven", fill=(226, 232, 238))
                dr.text((W + 10, frame.shape[0] - 16), "Chase camera", fill=(226, 232, 238))
                frame = np.asarray(im)
            writer.append_data(frame)
        writer.close()
        print(f"[video] wrote {args.out}")

    s = run.summary()
    s.update(scan=name, route_cm=[[round(x * 100, 1), round(y * 100, 1)] for x, y, _ in plan["path_xy"]], log=run.log)
    out_json = os.path.splitext(args.out)[0] + ".json"
    with open(out_json, "w") as f:
        json.dump(s, f)
    print(f"[sim]   {s['status']} in {s['sim_time_s']} s · driven {s['driven_cm']} cm · "
          f"max tilt {s['max_tilt_deg']} deg · max tracking error {s['max_tracking_error_cm']} cm")
    print(f"[done]  {time.time() - t0:.1f} s wall · log in {out_json}")


if __name__ == "__main__":
    main()
