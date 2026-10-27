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
| Secrets / env vars | None for the risk check. The optional AI dietary guidance needs `DIET_AGENT_MODEL` plus a provider API key (see section 5). `PATIENT_LOG_API_URL` and `DB_*` exist only for the optional local logger. |
| Extra processes | The MCP server is not a hosted service: CrewAI starts `src/mcp_server.py` as a short-lived local subprocess (stdio) when a plan is requested. Nothing listens on the network. |
| Local files needed at runtime | `models/best_model.joblib` and the `src/` modules. The training CSV is not needed. |
| Python / dependencies | Python 3.12; `requirements.txt` pins the model-critical versions (NumPy 2.5.3 needs Python 3.12+) and `crewai`/`mcp`. CrewAI adds roughly 100 transitive packages that are not pinned. |
| Health check | Streamlit's built-in `/_stcore/health` (returns `ok`). |

So the public service is a single Streamlit process. The AI guidance is the only part that talks to anything
outside it (an LLM provider), and only when a model and key are configured and a visitor asks for a plan.

## 2. Hosting options compared

| | **Streamlit Community Cloud** (recommended) | **Render** (alternative) | Hugging Face Spaces |
|---|---|---|---|
| Fit for this repo | Built for exactly this: pick repo, branch, `Streamlit/ui.py` | Works; you supply build/start commands | Poor fit (see below) |
| Cost | Free | Free web service tier (paid plans available) | Free tier cannot run this app |
| Deploy from GitHub | Yes, native | Yes (connect GitHub account for auto-deploy) | Separate Space git repo |
| Sleeps when idle | After 12 hours without traffic; anyone can wake it with one click | After 15 minutes; ~1 minute to wake | n/a |
| Resources | Approx. 0.078 to 2 CPU cores, 690 MB to 2.7 GB RAM (documented figures date from Feb 2024) | 512 MB RAM, 0.1 CPU | 2 vCPU / 16 GB, but see "Cost" |
| Config you must write | None (no start command, no port) | Build + start command, health check path | n/a |
| Secrets | "Advanced settings" / app Secrets (only for AI guidance) | Environment settings (only for AI guidance) | n/a |
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
free services are not for production. The risk check alone used roughly 190 MB resident in local testing
(Windows). With AI guidance enabled, the app process peaked at about 350 MB and about 420 MB including the
MCP subprocess while writing one plan, so 512 MB leaves little headroom for several visitors: on Render's
free tier leave AI guidance off (or expect restarts), or use a larger paid instance.

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
| 1 | Page loads, nothing in the terminal except startup lines | Title, the one-line "educational demo" caption, form |
| 2 | Click **Show result** with nothing filled in | "Please answer every question to see a result. Missing: ..." (no traceback) |
| 3 | Sample A (below) | "Higher-risk pattern" message, no percentage |
| 4 | Sample B (below) | "Lower-risk pattern" message saying it does not rule out diabetes |
| 5 | Type `200` (or `5`) into BMI and submit | The widget does not accept values outside 12 to 98, so the app lists BMI as unanswered; nothing crashes |
| 6 | Look at the terminal | No submitted values appear in the logs |
| 7 | Look for repeated disclaimer banners and the "How to read this result" expander | None: only the caption under the title remains |
| 8 | Stop the server, delete nothing, restart | Works again (no database, no state). The terminal shows `Serving diabetes-risk-classifier version 2` once, after the first result |
| 9 | `git status` shows no `mlflow_store/`, `mlruns/`, `*.db`, `*.log` files (the local MLflow store is git-ignored) | No drift or retraining code; MLflow is only a development tool |
| 10 | With `DIET_AGENT_MODEL` unset, get a result | An "AI dietary guidance" section says it is not set up; the risk check is unaffected |
| 11 | With a model and key set, get a result and click **Generate dietary guidance** (box unticked, then ticked) | A boxed "AI-generated" plan with its own warning appears below the result. With a bad key or no credit you get a friendly message, never a traceback |
| 12 | Look at the terminal during step 11 | Only CrewAI's "MCP Connection" panels (when the box is ticked); no answers, preferences or plan text |

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

Your CrewAI + MCP work is already committed (`c91b42d`) and on `origin/main`. The commands below review and
commit the changes made since then: MLflow tracking and the model registry (`src/tracking.py`,
`src/registry.py`, `src/model_card.py`, `models/model_card.json`), the diet-plan error handling, and the UI
cleanup. Make sure `models/model_card.json` is committed together with `models/best_model.joblib`: the app
refuses a model file that does not match its card.

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
git commit -m "Add MLflow tracking and model registry, fix AI plan errors, remove repeated disclaimers"
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
   have to delete and redeploy). Do not add `PATIENT_LOG_API_URL`. Secrets: leave empty to deploy the risk
   check only. To enable the AI dietary guidance, paste this (TOML) and use a spending-capped key:

   ```toml
   DIET_AGENT_MODEL = "openai/<model name>"
   OPENAI_API_KEY = "<your key>"
   ```

   Only top-level keys (above any `[section]` header) become environment variables, which is what the code
   reads. Pick a model name your provider currently lists as available and not deprecated: a retired
   model name makes every plan fail. Other providers need their CrewAI extra added to
   `requirements.txt` (for example `crewai[anthropic]`, `crewai[google-genai]`) and their own key variable.
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
- [ ] The "educational demo" line under the title is visible without scrolling; no "Deploy" button or developer menu.
- [ ] Submit with nothing filled in: friendly "Please answer every question" message.
- [ ] Sample A gives "Higher-risk pattern"; sample B gives "Lower-risk pattern"; no percentage anywhere.
- [ ] Works on a phone-sized window.
- [ ] Hosting logs show startup lines only: no health answers, no stack traces.
- [ ] Reloading and resubmitting works; nothing from a previous visitor appears.
- [ ] AI guidance: without secrets the section says it is not set up. With secrets, one test plan appears in
      its own box with the AI-generated warning, and your provider dashboard shows exactly one request.
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
| Build is slow or fails installing `crewai` dependencies | CrewAI pulls about 100 packages (chromadb, onnxruntime, openai...). Retry the build; if it keeps failing, delete the `crewai`, `mcp` and `pysqlite3-binary` lines from `requirements.txt` (keep `pydantic`) and the app deploys without AI guidance. |
| Logs show `unsupported version of sqlite3` | chromadb (imported by crewai) needs SQLite 3.35+. `pysqlite3-binary` in `requirements.txt` is the fallback and is applied automatically on Linux; confirm it installed. |
| AI section says "not set up" / "not installed" | `DIET_AGENT_MODEL` or the provider key is missing from the secrets, or `crewai` was not installed. The log has one line naming what is missing. |
| AI section says "not working on this server right now" | The provider answered 400, 401, 403 or 404: wrong or revoked key, a model name the account cannot use or that was retired, or a parameter the model rejects. Open **Manage app** to see the log line `AI provider call failed: ... (HTTP status N, provider code ..., parameter ...)`; the status and code say which. |
| AI section says the service is busy / over its usage limit | HTTP 429: rate limit or no credit left on the provider account. |
| AI section says the tool that reads the risk model could not be started | The local MCP subprocess failed to start in time (only when "share my answers" is ticked). Plans without sharing still work. |
| App says the prediction model is unavailable after you retrained | `models/best_model.joblib` no longer matches `models/model_card.json`. Run `python src/registry.py promote <version>` (or restore the committed file) and commit both. |
| Logs show an MCP connection timeout | CrewAI waits 10 s for the local MCP server to start. Schema-only mode starts in about 1.5 s here; `--with-prediction` loads the model first and can be slower. The app never uses that mode. |
| App "asleep" | Expected on free tiers; visitors wake it (Community Cloud) or the first request does (Render). |

## 9. Costs and free-tier limits

- **Streamlit Community Cloud:** free. Sleeps after 12 h without traffic; roughly 0.078 to 2 CPU cores and
  690 MB to 2.7 GB RAM (figures dated Feb 2024); US-only hosting; no SLA; one private app at a time.
- **Render free web service:** $0; 750 instance hours per calendar month; 512 MB RAM / 0.1 CPU; spins down
  after 15 minutes idle (about 1 minute to wake); ephemeral disk; Render may restart it at any time;
  bandwidth and build-minute allowances apply (see your dashboard's billing page). Paid plans: see
  <https://render.com/pricing>.
- **Hugging Face Spaces:** Streamlit/Docker apps need a paid plan (PRO is $9 per month).
- **LLM usage (only if AI guidance is enabled):** billed by your provider, not the host. One plan is about
  1,000 tokens in and 500 out (the stub run sent a ~4,000-character request), so at most a few thousand
  tokens per click; the app allows 3 plans per browser session by default (`DIET_AGENT_MAX_PER_SESSION`).
  That limit is per session, so set a monthly spending cap on the provider key as the real safeguard.
- If traffic grows, the practical next step is a paid always-on instance (for example a paid Render web
  service); the risk check itself is lightweight.

### What can run where (checked against the providers' own docs on 2026-10-25)

Free-tier terms change without notice; re-read the linked pages before relying on any of this.

| Part | Needs hosting? | Reasoning |
|---|---|---|
| Streamlit app | Yes | Community Cloud is free; apps with no traffic for 12 hours sleep and wake on a visit ([docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)). Nothing the app writes to disk should be relied on. |
| Prediction API (FastAPI) | No | The app predicts in-process. The existing FastAPI app only validates and logs to Postgres, and is off. A separately hosted API would add a network hop, a second service that sleeps, and nothing the page needs. |
| MCP server | No | It is a stdio subprocess started inside the app's own container for each plan that shares answers. It is not meant to be reachable from outside. |
| MLflow tracking + registry | No, keep it local | The deployed app never imports MLflow; it serves the committed `models/best_model.joblib` and `model_card.json`. MLflow is a development record. A remote one needs a database and artifact storage (MLflow supports SQLite, PostgreSQL, MySQL and MSSQL as the backend store and S3-style storage for artifacts: [docs](https://mlflow.org/docs/latest/self-hosting/architecture/overview)). |
| Scheduled / background jobs | Not on the free web hosts | Render's free tier is web services only: "Other service types don't support Free instances", its free web services spin down after 15 minutes without traffic and lose local files on restart, and free Postgres expires 30 days after creation ([docs](https://render.com/docs/free)). Hugging Face free Spaces sleep when unused and have non-persistent disk ([docs](https://huggingface.co/docs/hub/spaces-overview)). A web app being deployed does not mean a background process keeps running. There is no scheduled work in this project now. |

Options if you later want a shared, always-reachable MLflow (none was created or tested here):

- **Databricks Free Edition:** no-cost, quota-limited, non-commercial use only, and Databricks may delete
  accounts that are inactive for a prolonged period ([limits](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)).
  Those pages do not mention MLflow or a model registry, so confirm that before planning around it.
- **DagsHub free Individual plan:** 20 GB storage, up to 100 tracked experiments in private repositories,
  unlimited for public ones; its pricing page lists "MLflow Compatible" but does not make clear whether
  that is on the free plan ([pricing](https://dagshub.com/pricing)).
- **Self-host an MLflow server** on a web host with a managed Postgres: Neon's free plan has 1 GB
  per project and suspends compute after 5 minutes idle ([plans](https://neon.com/docs/introduction/plans)),
  but you also need somewhere for artifacts and a host for the server process, so it is not free of effort or
  of risk of cost.
- **GitHub Actions** can run a scheduled job: standard runners are free for public repositories, private
  repositories get 2,000 minutes per month on the Free plan, schedules can be delayed, and in a public
  repository they are disabled after 60 days without repository activity
  ([billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions),
  [events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)).

## 10. Privacy, security and regulatory notes

What happens to a visitor's answers: they travel from the browser to the host over HTTPS, are held in the
Streamlit session in server memory, produce a result, and go nowhere else. The app code writes them to no
database or file and logs nothing about them (automated tests assert submitted values never reach the logs
or any outbound request). The one exception is AI dietary guidance, which you enable deliberately: when a
visitor clicks Generate, their typed preferences (and, only if they tick the box, their result plus blood
pressure, cholesterol, BMI, general health and age group; never income or education) are sent to the LLM
provider you configured. The page says so before the button is pressed, the app stores none of it, CrewAI's
telemetry and tracing are forced off, and the MCP server runs only as a local subprocess. The provider's
own retention and training policies then apply, so choose one whose terms you have read, and mention it in
your privacy notice. Hosting platforms still keep ordinary web metadata such as IP addresses and
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
  in 10 who did were not flagged; the data is 2015 self-reported US survey data. The result wording
  ("does not rule out diabetes", "not a diagnosis") and the educational-demo line under the title are written
  to reflect that. Keep them.
