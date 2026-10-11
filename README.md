---
title: Tabletop Traverse
emoji: 🛰️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Tabletop Traverse

Built at [Sundai](https://www.sundai.club) · Autonomous Spacecraft Hack.

An iPhone LiDAR scan of a tabletop becomes rover terrain. The app maps hazards, plans a safe route from A to B, and drives a 4-wheel rover along it in a MuJoCo physics simulation. You can watch the run in 3D in the browser.

```
scan (.ply) → table plane (RANSAC) → 1 cm height grid → hazard classes → A* route
                                            │                               │
                                            ▼                               ▼
                               MuJoCo heightfield  ◄────────  pure-pursuit skid-steer rover
                                            │
                                            ▼
                         rover poses at 30 fps → three.js replay in the browser
```

## Run it locally

```bash
pip install -r requirements.txt
uvicorn server:app --port 7860
# open http://localhost:7860
```

A full scan → map → plan → simulate cycle takes about 1–2 seconds on a laptop.

## Deploy to Hugging Face Spaces (free, public URL)

1. Create a Space at https://huggingface.co/new-space. Choose **Docker** as the SDK, the **Blank** template, and the free CPU hardware.
2. Push this folder to it:
   ```bash
   git init && git add . && git commit -m "Tabletop Traverse"
   git remote add space https://huggingface.co/spaces/<your-username>/tabletop-traverse
   git push space HEAD:main        # password = a Hugging Face access token with write access
   ```
3. Wait for the build (about 3–5 minutes). The app is live at `https://<your-username>-tabletop-traverse.hf.space`.

The YAML block at the top of this README is the Space config. Keep it.

**Fallback:** if the Space build fails, run the server locally and share it with `cloudflared tunnel --url http://localhost:7860` or `ngrok http 7860`.

## Command line (video export, live viewer)

```bash
python rover_sim.py --scan scans/our_table.ply              # writes rover_run.mp4 + .json
python rover_sim.py --scan scans/our_table.ply --penalty 0  # drive over small objects
mjpython rover_sim.py --view                                # live MuJoCo viewer (macOS)
```

## Files

| File | What it does |
|---|---|
| `rover_core.py` | Scan loading, plane fit, hazard map, A*, MuJoCo world, rover controller |
| `server.py` | FastAPI: `/api/scan`, `/api/map`, `/api/simulate` |
| `static/index.html` | Frontend: hazard map with click-to-set A/B, 3D replay (three.js, vendored) |
| `rover_sim.py` | Command line: video export and live viewer |
| `scans/our_table.ply` | Our Scaniverse scan (decimated to 250k points) |

## How it maps to real rovers

| Here | On Mars |
|---|---|
| iPhone LiDAR point cloud | Stereo navcam / orbital (HiRISE) elevation maps |
| Step height, tilt and roughness grid | GESTALT / ENav traversability maps |
| Obstacle inflation by rover radius | Configuration-space planning |
| A* on a cost grid | Path search over the hazard map |
| MuJoCo drive with pure pursuit | Checking the plan against physics before committing |
