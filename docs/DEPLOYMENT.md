# Deployment

Backend on **Hugging Face Spaces** (Docker), frontend on **Vercel**. Both free,
no credit card, no time limit.

Everything here is configured and verified. Do not run it until you have a model
trained on real data — see the gate below.

---

## The gate: do not deploy a synthetic model

A public URL that accepts a chest X-ray and returns "Pneumothorax 78%, 95% CI
74–82%" in a clinical-looking report is a different object from the same code on
your laptop. Whoever finds it has no way to know the model was trained on
generated circles.

The pipeline now defends this itself: a checkpoint trained under `data/demo/` is
stamped `SYNTHETIC-DEMO-…`, and that string is what `/api/v1/health`, the results
panel and the PDF footer all display. But the right answer is not to rely on a
label — it is to deploy a model that has actually seen radiographs.

**Before deploying, confirm:**

```bash
python -c "
from radiscan.models.network import load_model
_, m = load_model('models/checkpoints/best_model.pth')
v = m.get('model_version', '?')
print('model_version:', v)
print('metrics      :', m.get('metrics'))
assert not str(v).startswith('SYNTHETIC'), 'This is a demo checkpoint. Do not deploy it.'
print('OK to deploy')
"
```

---

## Part 1 — Backend on Hugging Face Spaces

### 1.1 Create the Space

1. Sign up at [huggingface.co](https://huggingface.co) (free).
2. **New** → **Space**.
3. Name `radiscan-api`, licence **MIT**, SDK **Docker** → *Blank*, visibility
   **Public**.

### 1.2 Prepare the files

A Space is a git repo that needs the Dockerfile at its root, and a `README.md`
whose YAML frontmatter configures the Space.

```bash
git clone https://huggingface.co/spaces/<your-username>/radiscan-api hf-space
cd hf-space

# Copy the application in
cp -r ../radiscan/src ../radiscan/services ../radiscan/configs .
cp ../radiscan/pyproject.toml ../radiscan/requirements.txt ../radiscan/conftest.py .
cp ../radiscan/deploy/Dockerfile.spaces ./Dockerfile
cp ../radiscan/deploy/SPACE_README.md ./README.md
cp ../radiscan/.dockerignore .

# The checkpoint is ~32 MB, so it needs git-lfs
mkdir -p models/checkpoints
cp ../radiscan/models/checkpoints/best_model.pth models/checkpoints/
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

### 1.3 Set the secrets

In the Space: **Settings** → **Variables and secrets**.

| Name | Type | Value |
|---|---|---|
| `SECRET_KEY` | **Secret** | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `CORS_ORIGINS` | Variable | `https://<your-vercel-app>.vercel.app` |
| `ALLOW_UNTRAINED` | Variable | `false` |

`SECRET_KEY` must be a **Secret**, not a Variable — Variables are visible to
anyone who opens the Space page. Anyone holding that key can mint valid JWTs.

`CORS_ORIGINS` is a chicken-and-egg problem: you do not know the Vercel URL yet.
Deploy the frontend first, or come back and set it after (the Space rebuilds in
about a minute).

### 1.4 Push

```bash
git add -A
git commit -m "Deploy RadiScan API"
git push
```

First build takes 5–10 minutes, mostly PyTorch. Watch the **Logs** tab.

### 1.5 Verify

```
https://<your-username>-radiscan-api.hf.space/api/v1/health
```

Check that `model_loaded` is `true`, and that `model_version` does **not** start
with `SYNTHETIC`. Swagger is at `/docs`.

---

## Part 2 — Frontend on Vercel

### 2.1 Import

1. Sign in to [vercel.com](https://vercel.com) with GitHub.
2. **Add New** → **Project** → import `vanshvj01/RadiScan-AI`.
3. **Root Directory: `web`** ← the one setting that is easy to miss. Vercel
   defaults to the repo root, where there is no `package.json`, and the build
   fails with a confusing error.
4. Framework preset should auto-detect as **Next.js**.

### 2.2 Environment variable

| Name | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | `https://<your-username>-radiscan-api.hf.space/api/v1` |

Note the `/api/v1` suffix, and no trailing slash.

`NEXT_PUBLIC_` variables are baked in at **build** time, not read at runtime.
Changing it later requires a redeploy, not just a restart.

### 2.3 Deploy, then close the loop

Deploy, then take the Vercel URL back to the Space's `CORS_ORIGINS` and let it
rebuild. Until you do, every request from the browser fails with a CORS error
while `curl` against the API works perfectly — which is a confusing few minutes
if you are not expecting it.

---

## Known limitations of this setup

State these in your report rather than letting an examiner find them.

**Storage is ephemeral.** Free Spaces have no persistent disk. Accounts,
uploads, predictions and reports are wiped on every restart, and Spaces sleep
after ~48 h of inactivity. Fine for a demo; it is why the compose file exists
for a real deployment.

**Cold starts.** A sleeping Space takes 30–60 s to wake. Open the URL a few
minutes before you demo.

**CPU inference.** 20 MC-Dropout passes plus Grad-CAM on 2 vCPUs is roughly
5–15 s per image. The WebSocket progress channel exists precisely because of
this.

**No rate limiting.** The API has none. Public and free is fine for a project
demo; it would not be for anything real.

**Single worker.** One uvicorn process. Concurrent uploads queue.

---

## Redeploying after a change

**Backend:** copy the changed files into `hf-space/`, then `git push`. The Space
rebuilds automatically.

**Frontend:** `git push` to GitHub. Vercel rebuilds on every push to `main`, and
gives every branch its own preview URL.

**New model:** replace `models/checkpoints/best_model.pth` in the Space repo and
push. Run the gate check first — the label the pipeline stamps is a backstop,
not a substitute for looking.

---

## Local production check

Before pushing, confirm the image builds and boots:

```bash
docker build -f deploy/Dockerfile.spaces -t radiscan-api .
docker run -p 7860:7860 -e SECRET_KEY=local-test radiscan-api
curl localhost:7860/api/v1/health
```

If it works here it will work on Spaces — the container is the same.
