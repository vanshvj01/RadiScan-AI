# Run RadiScan locally, then deploy it

All commands are **PowerShell**, starting from the project folder:

    cd C:\Users\vansh\Downloads\radiscan-repo\radiscan

---

# Part A: Run locally (about 2 minutes)

You need **two PowerShell windows**: one for the backend and one for the dashboard.

### Window 1: backend (FastAPI on port 8000)

    cd C:\Users\vansh\Downloads\radiscan-repo\radiscan
    .venv\Scripts\Activate.ps1
    uvicorn services.api.main:app --reload

Wait for `Application startup complete`. Then check:

- http://localhost:8000/api/v1/health shows the model status
- http://localhost:8000/docs is the interactive API (Swagger)

It reads `SECRET_KEY` and the other settings from your `.env` file, which you already have.
Locally it loads `models\checkpoints\best_model.pth`. That is the **demo** model,
and the dashboard labels every result `SYNTHETIC-DEMO`. That is fine on your
laptop. Just say so if you demo it.

### Window 2: dashboard (Next.js on port 3000)

    cd C:\Users\vansh\Downloads\radiscan-repo\radiscan\web
    npm install        # first time only
    npm run dev

Open **http://localhost:3000**, register an account, log in and upload an X-ray.

### If something fails

| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded ... running scripts is disabled` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again |
| `npm ERR! enoent ... package.json` | You are in the wrong folder. `npm` must run inside `web\` |
| `uvicorn` not recognised | The venv is not active. Run `.venv\Scripts\Activate.ps1` |
| Dashboard loads but login fails / "Failed to fetch" | The backend window is not running, or it crashed. Check window 1 |
| Port 8000 or 3000 in use | Close the old window, or run `uvicorn ... --port 8001` and `$env:NEXT_PUBLIC_API_URL="http://localhost:8001/api/v1"` before `npm run dev` |
| Upload returns 503 | No checkpoint at `models\checkpoints\best_model.pth` |

Stop either server with **Ctrl + C**.

### Optional: run the tests

    .venv\Scripts\Activate.ps1
    pytest            # expect 87 passed

---

# Part B: Deploy (backend on Hugging Face, dashboard on Vercel)

```
Browser ──> Vercel (dashboard, web/) ──HTTPS──> Hugging Face Space (FastAPI + PyTorch)
```

Vercel cannot host the backend: PyTorch alone is far over Vercel's
function size limit. Hugging Face Spaces runs it in Docker for free.

## Two phases, because the model is not trained yet

| Phase | When | What the public site does |
|---|---|---|
| **1. Infrastructure** | Now | Everything works: sign-up, login, dashboard, history, API docs. Uploads return *"model not loaded"*. |
| **2. Model** | After the Kaggle CheXpert run | Copy in `best_model.pth`, push, done. Real predictions. |

**Do not upload the demo checkpoint to the public Space.** It was trained on
generated shapes. A public page that returns "Pneumothorax 78%" from it looks
real to anyone who finds it, whatever the label says. Phase 1 without a model
is honest. The demo model is not. (`ALLOW_UNTRAINED` must stay `false` for the
same reason.)

---

## Step 1: Push your code to GitHub

Vercel builds from GitHub, so it must have your latest code.

    git add paper tests\test_paper_fill.py notebooks\02_train_on_kaggle.ipynb docs\RUN-AND-DEPLOY.md
    git commit -m "Add paper, deck, Kaggle notebook and deployment guide"
    git push

## Step 2: Create the Hugging Face Space

1. Sign up at **huggingface.co** (free).
2. Top right **New → Space**.
3. Name **`radiscan-api`**. License **MIT**. SDK **Docker → Blank**. Hardware **CPU basic (free)**. Visibility **Public**.
4. Click **Create Space**.

## Step 3: Get a Hugging Face access token

**Settings → Access Tokens → Create new token**, type **Write**. Copy it.
You will paste it as the **password** in step 4. Never paste it into a chat or a file.

## Step 4: Put the backend into the Space

From the **parent** folder, so the Space repo sits next to your project:

    cd C:\Users\vansh\Downloads\radiscan-repo
    git clone https://huggingface.co/spaces/<your-hf-username>/radiscan-api hf-space
    cd hf-space

    Copy-Item -Recurse ..\radiscan\src, ..\radiscan\services, ..\radiscan\configs .
    Copy-Item ..\radiscan\pyproject.toml, ..\radiscan\requirements.txt, ..\radiscan\conftest.py, ..\radiscan\.dockerignore .
    Copy-Item ..\radiscan\deploy\Dockerfile.spaces .\Dockerfile
    Copy-Item ..\radiscan\deploy\SPACE_README.md .\README.md -Force

    # The Dockerfile copies models/, so the folder must exist even with no model yet
    New-Item -ItemType Directory -Force models\checkpoints | Out-Null
    New-Item -ItemType File models\checkpoints\.gitkeep | Out-Null

    git add -A
    git commit -m "Deploy RadiScan API (phase 1: no model)"
    git push

When git asks: **username** = your HF username, **password** = the token from step 3.

## Step 5: Set the Space's secrets

In the Space, open **Settings → Variables and secrets**.

Generate a key:

    python -c "import secrets; print(secrets.token_urlsafe(48))"

| Name | Add as | Value |
|---|---|---|
| `SECRET_KEY` | **New secret** | the generated key |
| `ALLOW_UNTRAINED` | New variable | `false` |
| `CORS_ORIGINS` | New variable | `http://localhost:3000` for now. You'll replace it in step 8. |

`SECRET_KEY` **must be a Secret**, not a Variable. Variables are public on the
Space page, and anyone holding the key can forge logins.

## Step 6: Wait for the build, then check it

The **Logs** tab shows the build. The first one takes **5 to 10 minutes** (mostly installing PyTorch).
When it says *Running*, open:

    https://<your-hf-username>-radiscan-api.hf.space/api/v1/health

Phase 1 is correct if you see `"database":"ok"` and `"model_loaded":false`.
`/docs` on the same address gives the Swagger UI.

## Step 7: Deploy the dashboard on Vercel

1. **vercel.com** → sign in with GitHub → **Add New → Project** → import **RadiScan-AI**.
2. **Root Directory: `web`**. This is the step people miss. Without it the build fails.
3. Framework: **Next.js** (auto-detected). Leave the build settings alone.
4. **Environment Variables**, add:

   | Name | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://<your-hf-username>-radiscan-api.hf.space/api/v1` |

   Use exactly that: `/api/v1` at the end, **no trailing slash**.
5. **Deploy**. You get a URL like `https://radiscan-ai.vercel.app`.

## Step 8: Connect the two (CORS)

The browser blocks the dashboard from calling the API until the API trusts
the dashboard's address. Back in the Space, edit the variable:

| Name | Value |
|---|---|
| `CORS_ORIGINS` | `https://radiscan-ai.vercel.app,http://localhost:3000` |

Use your real Vercel URL, with no trailing slash. The Space restarts in about a minute.

## Step 9: Test the live site

Open your Vercel URL. Register, log in, open the dashboard. In phase 1 an upload should
fail with a *no checkpoint* message (HTTP 503). That is correct.

### Deployment troubleshooting

| Symptom | Cause / fix |
|---|---|
| Vercel build: *Couldn't find package.json* | Root Directory is not `web`. Project → Settings → General → fix it, then redeploy |
| Live site: login says *Failed to fetch*; Chrome console says *CORS* | `CORS_ORIGINS` is missing your exact Vercel URL (check `https`, no trailing `/`) |
| Changed `NEXT_PUBLIC_API_URL` but nothing changed | It is baked in at build time. Vercel → Deployments → **Redeploy** |
| Space stuck *Building* / *Runtime error* | Read the Logs tab. `COPY models` failing means step 4's `models\checkpoints` folder is missing |
| First request takes ~60 s | Free Spaces sleep after inactivity. Open the site 5 minutes before you present |

---

## Phase 2: add the trained model (after Kaggle)

1. Put the Kaggle `best_model.pth` in `radiscan\models\checkpoints\`.
2. Run the gate check from the project folder. It must print **OK to deploy**:

       python -c "from radiscan.models.network import load_model; _, m = load_model('models/checkpoints/best_model.pth'); v = m.get('model_version','?'); print(v); assert not str(v).startswith('SYNTHETIC'), 'DEMO checkpoint - do not deploy'; print('OK to deploy')"

3. Add it to the Space with Git LFS (it is about 32 MB):

       cd C:\Users\vansh\Downloads\radiscan-repo\hf-space
       git lfs install
       git lfs track "*.pth"
       Copy-Item ..\radiscan\models\checkpoints\best_model.pth models\checkpoints\
       git add .gitattributes models
       git commit -m "Add CheXpert-trained model"
       git push

4. After the rebuild, `/api/v1/health` must show `"model_loaded":true` and a
   `model_version` that does **not** start with `SYNTHETIC`. Uploads now return
   real predictions. Expect 5 to 15 seconds per image on the free CPU.

## Known limits (tell the panel; don't let them discover these)

- **Storage resets**: accounts and scans are wiped when the free Space restarts or sleeps.
- **CPU only**: 20 Monte Carlo passes take seconds, not milliseconds.
- **No rate limiting**, single worker: a demo deployment, not a clinical one.
- **Not a medical device**: the UI and PDF reports say so.
