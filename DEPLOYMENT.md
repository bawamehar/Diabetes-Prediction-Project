# Deployment guide

How to put the Diabetes Risk Check on the public internet. Nothing in this repository deploys itself:
every step below is manual and done by you.

Platform details were checked against the providers' official documentation on **2026-10-22**. Free tiers
change often, so re-check the linked pages before you rely on a specific number.

## 1. What actually has to be hosted

| Question | Answer (from the code) |
|---|---|
| Frontend / entry point | Streamlit, `Streamlit/ui.py` |
| Backend / prediction API | None needed. The app loads the model and predicts in-process (`src/inference.py`). |
| Model + preprocessing | One pickled scikit-learn `Pipeline` (scaler + classifier), `models/best_model.joblib` (~144 KB), loaded relative to the file, so any working directory works. |
| Database | **Not required.** The optional `api/` + PostgreSQL logger is off by default and must stay off on a public deployment. |
| Secrets / env vars | None required. (`PATIENT_LOG_API_URL` and `DB_*` exist only for the optional local logger.) |
| Local files needed at runtime | Only `models/best_model.joblib`. The training CSV is not needed. |
| Python / dependencies | Python 3.12; `requirements.txt` pins exact versions (NumPy 2.5.3 needs Python 3.12+). |
| Health check | Streamlit's built-in `/_stcore/health` (returns `ok`). |

So the public service is a single Streamlit process with no external dependencies.

## 2. Hosting options compared

| | **Streamlit Community Cloud** (recommended) | **Render** (alternative) | Hugging Face Spaces |
|---|---|---|---|
| Fit for this repo | Built for exactly this: pick repo, branch, `Streamlit/ui.py` | Works; you supply build/start commands | Poor fit (see below) |
| Cost | Free | Free web service tier (paid plans available) | Free tier cannot run this app |
| Deploy from GitHub | Yes, native | Yes (connect GitHub account for auto-deploy) | Separate Space git repo |
| Sleeps when idle | After 12 hours without traffic; anyone can wake it with one click | After 15 minutes; ~1 minute to wake | n/a |
| Resources | Approx. 0.078 to 2 CPU cores, 690 MB to 2.7 GB RAM (documented figures date from Feb 2024) | 512 MB RAM, 0.1 CPU | 2 vCPU / 16 GB, but see "Cost" |
| Config you must write | None (no start command, no port) | Build + start command, health check path | n/a |
| Secrets | Settings UI (not needed here) | Environment settings (not needed here) | n/a |
| HTTPS | Yes, on `*.streamlit.app` | Yes, managed TLS | n/a |
| Custom domain | Not documented; you can choose the `*.streamlit.app` subdomain name | Yes | n/a |
| Public without viewers needing an account | Yes, once the app is set to public | Yes | n/a |
| Can the source repo stay private? | Yes: deploy from a private repo, then make the *app* public | Yes (private repos can be connected) | Protected Spaces need a paid plan |
| Persistent storage / Postgres | None needed | Free disk is ephemeral; free Postgres expires after 30 days | n/a |
| Region | United States only | Choose a region | n/a |

**Why Streamlit Community Cloud for this repo.** The app is one Streamlit script with an in-memory
model, no database and no secrets, so there is nothing else to provision. Community Cloud deploys it from
a GitHub repository with no start command, port, or Procfile, installs from `requirements.txt`, defaults
to Python 3.12 (what this repo is pinned for), forces `showErrorDetails = false` and XSRF protection, and
is free. A private source repository can still back a public app.

Its real downsides for this app: the app sleeps after 12 hours without a visit (the first visitor sees a
"get this app back up" page), hosting is US-only, there is no SLA, and custom domains are not documented.

**Why Render is the alternative.** It runs the same app with ordinary `pip install` + a start command (no
Docker), supports custom domains with managed TLS, and could later also host the optional FastAPI service.
Its free tier is more fragile for a public demo: 512 MB RAM / 0.1 CPU, the service spins down after 15
minutes idle and takes about a minute to wake, 750 free instance hours per month, and Render's own docs say
free services are not for production. The app used roughly 190 MB resident in local testing (Windows),
so it should fit in 512 MB, but check Render's metrics after deploying.

**Why not Hugging Face Spaces.** Hugging Face's current docs offer Gradio, Docker and static SDKs; creating
Gradio or Docker Spaces needs a paid plan (PRO, $9 per month for personal accounts), and Streamlit is not
listed as its own SDK. Running this app there would mean adding Docker and paying, so it is not a free
option. Static Spaces are free but cannot run Python.

Other platforms (Fly.io, Railway, Google Cloud Run, etc.) were not evaluated against current
documentation. They generally want a container or paid plan for an app like this, so they are not
recommended for a first portfolio deployment.

Sources: [Streamlit Community Cloud docs](https://docs.streamlit.io/deploy/streamlit-community-cloud),
[Render free tier](https://render.com/docs/free), [Render Python version](https://render.com/docs/python-version),
[Render web services](https://render.com/docs/web-services),
[Hugging Face Spaces overview](https://huggingface.co/docs/hub/spaces-overview).

## 3. Before you push: verify locally

Run from the repository root (`F:\Diabetes Project\ClassificationHealthDiabetes`) in PowerShell.

```powershell
# 1. Environment (Python 3.12)
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# 2. Automated checks (expect all to pass; the API tests run because FastAPI is installed)
pytest

# 3. Make sure nothing is configured to store data
Get-ChildItem Env:PATIENT_LOG_API_URL     # should print nothing / "Cannot find"

# 4. Start the app the same way a host would (use any free port)
$env:PORT = "8501"
streamlit run Streamlit/ui.py --server.port $env:PORT --server.address 0.0.0.0 --server.headless true
```

Then, in a second terminal:

```powershell
Invoke-WebRequest http://localhost:8501/_stcore/health -UseBasicParsing | Select-Object -ExpandProperty Content   # ok
```

Open <http://localhost:8501> and walk through this checklist:

| # | Check | Expected |
|---|---|---|
| 1 | Page loads, nothing in the terminal except startup lines | Title, yellow disclaimer, form |
| 2 | Click **Show result** with nothing filled in | "Please answer every question to see a result. Missing: ..." (no traceback) |
| 3 | Sample A (below) | "Higher-risk pattern" message, no percentage |
| 4 | Sample B (below) | "Lower-risk pattern" message saying it does not rule out diabetes |
| 5 | Type `200` (or `5`) into BMI and submit | The widget does not accept values outside 12 to 98, so the app lists BMI as unanswered; nothing crashes |
| 6 | Look at the terminal | No submitted values appear in the logs |
| 7 | Open "How to read this result, and its limits" | Explains the limits and why no percentage is shown |
| 8 | Stop the server, delete nothing, restart | Works again (no database, no state) |
| 9 | `git status` shows no `mlflow.db`, `mlruns/`, `*.log` files | Drift/retraining workflow is fully gone |

Sample inputs (all values are inside the model's training ranges):

| | HighBP | HighChol | BMI | General health | Age group | Education | Income | Expected |
|---|---|---|---|---|---|---|---|---|
| A | Yes | Yes | 45 | Poor | 80 or older | Never attended school or only kindergarten | Less than $10,000 | Higher-risk pattern |
| B | No | No | 22 | Excellent | 30 to 34 | College 4 years or more | $75,000 or more | Lower-risk pattern |
| C | Yes | No | 32 | Good | 60 to 64 | Grade 12 or GED | $25,000 to less than $35,000 | Higher-risk pattern |

Invalid and missing input is also covered by automated tests (`tests/test_inference.py`, `tests/test_ui.py`).

Troubleshooting local runs:

- `ModuleNotFoundError` -> the venv is not active, or `pip install -r requirements.txt` was skipped.
- A scikit-learn `InconsistentVersionWarning` -> your installed version differs from `requirements.txt`.
- "The prediction model is currently unavailable" -> `models/best_model.joblib` is missing or unreadable.
- Port already in use -> pass a different `--server.port`.

## 4. GitHub: review, test, commit, push

Your working tree contains all the changes from this work **and** earlier uncommitted work (the "always
predicts non-diabetic" fix, LightGBM, the pipeline model). Review before committing.

```powershell
cd "F:\Diabetes Project\ClassificationHealthDiabetes"

# See what changed and what is new
git status
git diff --stat
git diff                                   # or: git diff -- Streamlit/ui.py src/inference.py
git status --short | Select-String '^\?\?' # new files that are not yet tracked

# Remove compiled .pyc files from version control (they are already tracked; .gitignore stops new ones)
git rm -r --cached --quiet src/__pycache__ api/__pycache__

# Optional but recommended: these two CSVs are from an unrelated Kaggle competition (calorie prediction),
# not this project, and competition data may not be redistributable. Remove after you confirm.
git rm Data/data/test.csv Data/data/sample_submission.csv

# An empty nested git repository sits at src/.git (no commits). Safe to delete; it can confuse git add.
Remove-Item src\.git -Recurse -Force

# Final tests
.\.venv\Scripts\Activate.ps1
pytest

# Secret / sensitive-data scan of tracked + new files. Expect hits only for the env-var NAMES
# (DB_PASSWORD) in api/ and the docs - never an actual credential value.
git grep -n -I -i --untracked -E "password|secret|token|api[_-]?key|BEGIN (RSA|OPENSSH|PRIVATE)" -- . ":(exclude)Notebooks" ":(exclude)uv.lock"
git ls-files --others --exclude-standard   # anything unexpected here? (no .env, no .log, no *.db)

# Dry run of what would be staged, then stage by name (avoid `git add .`)
git add -n .
git add README.md DEPLOYMENT.md .gitignore .streamlit requirements.txt requirements-dev.txt pyproject.toml `
        Streamlit api src tests models .github
git status

# Commit
git commit -m "Remove drift/retraining workflow; make the Streamlit app safe and ready for public deployment"
```

Push (the repository's current branch is `main`, tracking `origin/main`):

```powershell
git remote -v        # confirm it points at your repository (see note below)
git push origin main
```

Safer alternative: commit on a branch, push it, review on GitHub, and deploy from that branch first.

```powershell
git switch -c deployment-readiness
git commit ...       # as above
git push -u origin deployment-readiness
```

Note: `git remote -v` currently shows `https://github.com/bawamehar/ClassificationHealthDiabetes.git`, while
your repository page shows the name `Diabetes-Prediction-Project`. GitHub redirects renamed repositories, so
pushes still work, but you can update the URL with
`git remote set-url origin https://github.com/bawamehar/Diabetes-Prediction-Project.git` once you confirm
the new name.

### Should the repository be public?

A public deployment does **not** require a public repository: Streamlit Community Cloud can deploy from a
private repo and serve the app publicly. What the repository contains today:

- No credentials, tokens or private keys were found in the working tree or in the 15-commit history. The
  only credential-like value that existed (`"password"` as a default Postgres password) has been replaced
  by environment variables. Because it exists in older commits, never reuse that password anywhere.
- The training data (`Data/data/diabetes.csv`, ~23 MB) is a public CDC survey dataset. Confirm its license
  and acknowledgement terms on the dataset page before keeping it in a public repo.
- `Data/data/test.csv` and `sample_submission.csv` look like they belong to another Kaggle competition:
  remove them (command above).
- `Notebooks/` is tracked and contains saved outputs and a local absolute path in
  `Notebooks/mlruns/*/meta.yaml`. Harmless, but skim the notebooks for anything you would not want public.

If you are unsure, keep the repository private and deploy from it.

## 5. Deploy on Streamlit Community Cloud (recommended)

1. **Account.** Go to <https://share.streamlit.io> and sign in with GitHub. Authorize Streamlit to access
   your repositories. For a *private* repository, GitHub asks for the broader `repo` scope, and you need
   administrator permission on that repository.
2. **Repository check.** On the branch you will deploy, confirm these exist at the repo root:
   `requirements.txt`, `.streamlit/config.toml`, `models/best_model.joblib`, `Streamlit/ui.py`. Also confirm
   there is **no** `uv.lock`, `Pipfile` or `environment.yml` (Community Cloud picks the first it finds in
   this order: `uv.lock`, `Pipfile`, `environment.yml`, `requirements.txt`, `pyproject.toml`, so a stale
   `uv.lock` would override `requirements.txt`).
3. **Create the app.** In your workspace click **Create app** -> **Yup, I have an app**, then fill in:
   - Repository: `bawamehar/<your repository name>`
   - Branch: `main` (or your deployment branch)
   - Main file path: `Streamlit/ui.py` (capital S: Linux paths are case-sensitive)
   - App URL: choose a subdomain (6 to 63 characters), e.g. `diabetes-risk-check`
4. **Advanced settings.** Python version: **3.12**. This cannot be changed after deployment (you would
   have to delete and redeploy). Secrets: leave empty. Do not add `PATIENT_LOG_API_URL`.
5. **Deploy.** Click **Deploy**. The first build installs `requirements.txt` and usually takes a few minutes.
   Logs appear on the right side of the app page (visible only to people with write access to the repo).
6. **Make it public.** If the repository is private the app is private by default: open the app's
   **Share** button (or App settings -> Sharing) and set *Who can view this app* to
   **This app is public and searchable**. If the repository is public the app is public by default. Only
   one app at a time can be private on a free account.
7. **Validate** with the public-URL checklist in section 7.

Updating: push to the deployed branch; most changes appear almost immediately, and changing
`requirements.txt` triggers a longer reinstall. Use **Manage app -> Reboot app** if it misbehaves.
Updates from GitHub are rate-limited to five per minute.

Removing: open the app -> **Manage app** -> overflow menu -> **Delete app** (or use the overflow menu next
to the app in your workspace). Switching the app back to private also stops public access.

## 6. Deploy on Render (alternative)

1. **Account.** Sign up at <https://render.com> (GitHub login is simplest) and connect your GitHub account
   so Render can read the repository. Render's docs describe Free services running without a payment
   method: if usage exceeds the included allowance and no payment method is on file, Render suspends the
   free services for the rest of the month instead of billing you.
2. **New service.** Dashboard -> **New** -> **Web Service** -> choose your repository (not "public Git
   repository URL": auto-deploy is not supported for that mode).
3. **Settings:**

   | Field | Value |
   |---|---|
   | Branch | `main` (or your deployment branch) |
   | Language / runtime | Python 3 |
   | Root directory | leave empty |
   | Build command | `pip install -r requirements.txt` |
   | Start command | `streamlit run Streamlit/ui.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
   | Instance type | Free |
   | Health check path (Advanced) | `/_stcore/health` |
   | Environment variable | `PYTHON_VERSION` = `3.12.7` (full version required; Render otherwise also reads `.python-version`, which says `3.12`) |
   | Auto-deploy | On commit |

   Render requires the service to listen on `0.0.0.0` and recommends the `PORT` variable (default 10000),
   which the start command above uses. Do not add `DB_*` or `PATIENT_LOG_API_URL`.
4. **Create** and watch the **Logs** tab. A successful start prints "You can now view your Streamlit app".
5. Open the `https://<name>.onrender.com` URL and run the section 7 checklist. The first request after
   15 idle minutes takes about a minute while the service wakes.
6. Custom domain: **Settings -> Custom Domains** (managed TLS is included).

Updating: every push to the linked branch redeploys. Disabling: **Settings -> Suspend Service** (reversible)
or **Delete Service**.

## 7. Public-URL validation checklist

Open the URL in a private/incognito window, signed out of GitHub and the hosting provider.

- [ ] Loads over HTTPS with no login prompt.
- [ ] Disclaimer is visible without scrolling; no "Deploy" button or developer menu.
- [ ] Submit with nothing filled in: friendly "Please answer every question" message.
- [ ] Sample A gives "Higher-risk pattern"; sample B gives "Lower-risk pattern"; no percentage anywhere.
- [ ] Works on a phone-sized window.
- [ ] Hosting logs show startup lines only: no health answers, no stack traces.
- [ ] Reloading and resubmitting works; nothing from a previous visitor appears.
- [ ] (Free tiers) Come back after the idle period and confirm it wakes up.

## 8. Troubleshooting deployments

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: streamlit` / `sklearn` | The platform did not install `requirements.txt`. Check it is at the repo root, is UTF-8, and that no `uv.lock`/`Pipfile`/`environment.yml` is taking priority. |
| Build fails resolving NumPy | The platform is using Python older than 3.12. Select 3.12 (Community Cloud: delete and redeploy; Render: set `PYTHON_VERSION`). |
| scikit-learn `InconsistentVersionWarning` or unpickling error | Installed versions differ from the pins in `requirements.txt` (the model was pickled with scikit-learn 1.9.1). Keep the pins; if you retrain, update them. |
| "The prediction model is currently unavailable." | `models/best_model.joblib` is not in the deployed branch (check `git ls-files models`) or failed to unpickle (see logs). |
| Main file not found | The path is case-sensitive: `Streamlit/ui.py`. |
| Render: "No open ports detected" | The start command must use `--server.address 0.0.0.0 --server.port $PORT`. |
| Render: killed / out of memory | Free tier is 512 MB. Restart, or use Community Cloud (larger memory allowance). |
| Database connection errors | The app does not use a database. Make sure `PATIENT_LOG_API_URL` is not set in the hosting environment. |
| App "asleep" | Expected on free tiers; visitors wake it (Community Cloud) or the first request does (Render). |

## 9. Costs and free-tier limits

- **Streamlit Community Cloud:** free. Sleeps after 12 h without traffic; roughly 0.078 to 2 CPU cores and
  690 MB to 2.7 GB RAM (figures dated Feb 2024); US-only hosting; no SLA; one private app at a time.
- **Render free web service:** $0; 750 instance hours per calendar month; 512 MB RAM / 0.1 CPU; spins down
  after 15 minutes idle (about 1 minute to wake); ephemeral disk; Render may restart it at any time;
  bandwidth and build-minute allowances apply (see your dashboard's billing page). Paid plans: see
  <https://render.com/pricing>.
- **Hugging Face Spaces:** Streamlit/Docker apps need a paid plan (PRO is $9 per month).
- If traffic grows, the practical next step is a paid always-on instance (for example a paid Render web
  service); the app itself is lightweight.

## 10. Privacy, security and regulatory notes

What happens to a visitor's answers: they travel from the browser to the host over HTTPS, are held in the
Streamlit session in server memory, produce a result, and go nowhere else. The app code writes them to no
database or file and logs nothing about them (automated tests assert submitted values never reach the logs
or any outbound request). Hosting platforms still keep ordinary web metadata such as IP addresses and
request times under their own terms: read the privacy policy of whichever host you pick. Usage telemetry
from Streamlit itself is disabled in `.streamlit/config.toml`.

Things to review **before** opening the app to the public (this is not legal advice):

- **Sensitive data.** Health answers plus income and education are sensitive. Keep collection to
  nothing, as it is now, and do not add analytics, tracking, accounts or storage without a clear purpose
  and notice.
- **Privacy notice.** Add a short statement of what is and is not collected and who the host is.
- **Regulation.** Do not describe the app as "HIPAA-compliant" or otherwise "compliant": there is no
  evidence for that. Whether HIPAA, GDPR/UK GDPR (health data is a special category, and Community Cloud
  hosts in the US), US state consumer-health-data laws, or medical-device rules apply depends on who your
  users are and what you claim the tool does. Keeping the language educational, as the app does, helps;
  avoid words like "diagnose" or "detect".
- **Audience.** The survey covers adults (18+). The app does not target or verify minors.
- **Abuse.** Streamlit has no built-in rate limiting. Prediction is cheap and the inputs are bounded
  (fixed choices, numeric range, server-side validation, no free text or uploads), so abuse risk is low;
  platform rate limits and the free-tier resource caps are the practical backstop.
- **Secrets.** None are required. Never commit `.env` files or `.streamlit/secrets.toml` (both are in
  `.gitignore`).
- **Model limits.** Roughly 1 in 3 people flagged as higher risk reported diabetes/prediabetes and about 2
  in 10 who did were not flagged; the data is 2015 self-reported US survey data. The disclaimer and result
  wording are written to reflect that. Do not remove them.
