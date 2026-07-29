# Recipe: interactive GPU applets viewed from the laptop browser

Serve an interactive visualization from a GPU on the cluster (`l40-worker`) and open it
in the laptop's browser. Verified 2026-07: the login VM cannot reach the compute node at
all (neither SSH nor high ports — firewalled), but the node **can** SSH back to the
login VM (`10.0.4.210`). So the only working path is a **reverse tunnel**:

```
browser → laptop:PORT → (ssh -L) → login-VM:PORT → (ssh -R, from node) → node:PORT → applet
```

## One-time setup

Passwordless key for node → login-VM SSH (home is shared NFS, so one keypair covers
both ends). The `restrict,port-forwarding` prefix makes the key forward-only (no shell):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_tunnel -N "" -C cluster-tunnel
printf 'restrict,port-forwarding %s\n' "$(cat ~/.ssh/id_ed25519_tunnel.pub)" >> ~/.ssh/authorized_keys
```

## Writing the applet

- One process, one port, bound to `127.0.0.1` — the tunnel originates on the node
  itself, so localhost is enough (and safer on a shared node).
- FastAPI + uvicorn (already in the venv) serving an inline HTML page + JSON API is the
  simplest shape. See `toy_applet.py` here for a complete ~70-line example.
- Use **relative** URLs in the page's JS (`fetch("api/...")`, not
  `fetch("http://host:port/api/...")`) so the page works through any tunnel.
- Multi-port setups (e.g. the PD app's Vite frontend + FastAPI backend) work too: only
  the frontend port needs tunneling because Vite proxies `/api` internally.

## Per session

Pick a port (below: 8123). On the login VM, **inside tmux** (so a dropped laptop
connection doesn't kill the job):

```bash
srun --gres=gpu:l40:1 --time=8:00:00 --job-name=pd-applet --pty bash
```

Then on the node (venv is on shared NFS, already activated if you activated before srun):

```bash
ssh -i ~/.ssh/id_ed25519_tunnel -o StrictHostKeyChecking=accept-new \
    -N -R 8123:localhost:8123 10.0.4.210 &
python notes/gpu_applet/toy_applet.py --port 8123
```

On the laptop:

```bash
ssh -L 8123:localhost:8123 <login-vm>
```

Open <http://localhost:8123>.

## Gotchas

- GPU cap is 6 concurrent GPUs (see memory); the applet's GPU counts toward it.
- No-gres jobs still see all 8 GPUs (no device cgroup isolation on this cluster) —
  torch will silently grab an unallocated GPU. For a CPU-only plumbing test, request no
  gres AND `export CUDA_VISIBLE_DEVICES=""`; for real use, always request `--gres` so
  the GPU is accounted for.
- `-R` fails silently-ish ("remote port forwarding failed") if the login-VM port is
  already taken — pick another port, or `pkill -f 'ssh.*-R 8123'` a stale tunnel.
- First `ssh -R` from the node needs `StrictHostKeyChecking=accept-new` (host key not
  yet known there — actually shared NFS `known_hosts` makes this a one-time event).
- Sanity-check the chain without the laptop: `curl http://localhost:8123/` **on the
  login VM** exercises applet + reverse tunnel.
