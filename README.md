# RAG Resume Builder

Tailor your resume to any job post, and get proof for every line.

You upload your resume once. For each job post, the app finds the matching evidence in your own history and
writes bullets **only from that evidence**. Python then checks every bullet before it reaches the page. You get
a one-page PDF and an evidence map that shows which requirement each bullet proves, and which ones you
can't prove yet.

It runs on your machine. You bring one LLM API key: OpenAI, Anthropic or Google.

---

## Contents

- [Why this exists](#why-this-exists)
- [Features](#features)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Your profile](#your-profile)
- [API keys and providers](#api-keys-and-providers)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Command line](#command-line)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Privacy](#privacy)
- [Contributing](#contributing)
- [License](#license)

---

## Why this exists

Asking a chatbot to "tailor my resume to this job" usually goes wrong in one of two ways. It invents
experience you don't have, or it stuffs in keywords you can't back up in an interview.

This app takes the opposite approach:

- **Retrieval first.** Every bullet starts from text you wrote about your own work.
- **Checked, not trusted.** A bullet is rejected if it uses a job-post keyword that isn't in the cited
  evidence, or a number that doesn't come from it.
- **Honest gaps.** Requirements you can't prove are reported as gaps, not papered over.

## Features

| | |
|---|---|
| **Two ways in** | Upload a resume **PDF** (an LLM copies its facts into a structured profile) or a **JSON** profile you write yourself. JSON can hold far more than one page: every project, every detail. |
| **Hybrid retrieval** | Parent-child chunks in ChromaDB. BM25 and vector search are fused with RRF, with one query per requirement line. |
| **Planner** | Each requirement gets one home section (job, project or research) before any writing, so the best proof lands in the right place. |
| **Evidence-checked writing** | A LangGraph agent writes section by section and uses each piece of evidence once. A humanizer pass removes AI-sounding phrasing, then every bullet is validated again. |
| **Coverage report** | Each requirement is marked **PROVEN**, **MISSED**, **GAP** or **ROUTED**, with coverage and recall scores. |
| **One-page PDF** | A fit loop prints with headless Chrome or Edge until the resume fits one page. Edit bullets in the app and rebuild. |
| **Built-in PDF viewer** | Zoom, drag to pan, and full screen. |
| **Auto company and role** | Leave them blank and they're read from the post. When the LLM isn't sure, you pick from chips. |
| **Any one LLM key** | OpenAI, Anthropic or Google. Extra keys become automatic fallbacks. |
| **Cost tracking** | Every LLM call is priced with LiteLLM and recorded per job post. Repeat runs hit an exact-match cache and cost $0. |

## Quick start

### Requirements

- **Python 3.11+**
- **Node.js 18+**
- **Chrome, Edge or Chromium** installed, used to print the PDF
- **One API key** from [OpenAI](https://platform.openai.com/api-keys),
  [Anthropic](https://console.anthropic.com/) or [Google AI Studio](https://aistudio.google.com/apikey)

### 1. Install

```bash
git clone <this-repo-url> rag-resume-builder
cd rag-resume-builder

# backend
cd backend
pip install -r requirements.txt

# frontend
cd ../frontend
npm install
cd ..
```

A virtual environment is recommended (`python -m venv .venv`, then activate it) but not required.

### 2. Add a key

```bash
cp backend/.env.example backend/.env       # Windows: copy backend\.env.example backend\.env
```

Open `backend/.env` and fill in **at least one** of these:

```ini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
```

### 3. Run

```bash
./start.sh            # macOS / Linux
.\start.ps1           # Windows PowerShell
```

This starts the API on **:8710** and the web app on **:5180**. Open **http://localhost:5180**.

<details>
<summary>Running the two parts by hand</summary>

```bash
# terminal 1
cd backend && python server.py

# terminal 2
cd frontend && npm run dev
```

On Windows, if PowerShell refuses to run `start.ps1`, run
`Set-ExecutionPolicy -Scope Process Bypass` in that window first.
</details>

> **Changed `.env` or backend code?** Restart the backend. It reads your keys only at startup.

## Using the app

### Step 1: Add your resume

Open **Upload your resume** and drop in a file:

- **PDF:** one LLM call reads it into a structured profile. Fastest way to start.
- **JSON:** used as is. Best results, because you can include more than fits on one page.
  Click **Empty template** to start a JSON profile from scratch.

The profile is then split into chunks and indexed. You only do this once, or again after you edit your profile.

### Step 2: Tailor to a job

Click **New tailor** and paste the **full** job post (at least 200 characters), including the qualifications
and responsibilities.

**Company** and **Role** are optional:

- **Both filled in:** used as you typed them. No extra LLM call.
- **Left blank:** read from the post. A name is accepted automatically only when the LLM is confident
  **and** the exact text appears in the post.
- **Not clear from the post** (for example a recruiter post for an unnamed client): chips appear. Pick one,
  type your own, or choose **Not stated**, then click **Confirm and tailor**.

Click **Tailor resume**. The pipeline view shows each stage as it runs.

### Step 3: Review, edit, export

When the run finishes you get:

- **The resume,** section by section, with the evidence behind each bullet. Edit any line and save.
- **The coverage report:** which requirements are proven, missed or gaps.
- **The PDF panel:** zoom with the buttons or Ctrl + scroll, drag to pan when zoomed in, open full screen,
  click **Build** after edits, and **PDF** to download.

All past runs are listed in the sidebar. The cost icon in the header shows spend per job post.

## Your profile

Uploading a PDF creates `backend/data/profile.json`. You can download it from the app, fix anything the
extraction got wrong, add more detail, and upload the JSON again. A filled-in example lives at
[`examples/profile.example.json`](examples/profile.example.json).

```jsonc
{
  "personal": { "full_name": "", "email": "", "phone": "", "location": "",
                "links": { "linkedin": "", "github": "", "portfolio": "", "other": [] } },
  "summary": "",
  "skills": { "languages": ["Python"], "frameworks": ["React"] },
  "work_experience": [{ "company": "", "title": "", "location": "", "start": "", "end": "",
                        "description": "", "highlights": [], "tech": [] }],
  "projects": [{ "name": "", "description": "", "highlights": [], "tech": [], "live": "", "repo": "" }],
  "research_experience": [{ "title": "", "institution": "", "location": "", "start": "", "end": "",
                            "advisor": "", "description": "", "highlights": [], "tech": [] }],
  "education": [{ "institution": "", "location": "", "degree": "", "field": "", "gpa": "", "start": "", "end": "" }],
  "certifications": [{ "name": "", "issuer": "", "date": "" }]
}
```

- Missing keys are filled in for you.
- Extra keys you add (for example `"metrics"` or `"engineering_notes"` on a job) are kept and indexed as
  evidence too.
- **Tip:** the more concrete detail you include (numbers, tools, scope, outcomes), the more the agent can prove.
  It can only cite what's in the profile.

## API keys and providers

**One key is required. A second or third is optional.**

| Key | Writer model (default) | Embeddings |
|---|---|---|
| `OPENAI_API_KEY` | `openai/gpt-5.4` | `text-embedding-3-small` |
| `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-5-5` | local model (Anthropic has no embedding API) |
| `GOOGLE_API_KEY` | `gemini/gemini-2.5-pro` | `gemini-embedding-001` |

**How calls are routed**

1. Every LLM call goes to **one** provider: the first key found, checked in the order OpenAI, Anthropic, Google.
2. If that call fails (bad key, rate limit, outage), the **next** configured provider retries it.
3. If every provider fails, you get one error listing each failure.

Set `LLM_PROVIDER=anthropic` (or `openai` / `google`) to put a different provider first.

**Embeddings** use OpenAI or Gemini, whichever comes first. With only an Anthropic key, a small local model
(ChromaDB's built-in `all-MiniLM-L6-v2`) is used. It downloads once (about 80 MB) and costs nothing to run.

> **If you change keys, re-index.** Vectors from different embedding models can't be mixed. If your keys now
> select a different embedding model, the app stops and asks you to re-index. Click **Re-index** in the app or
> run `python backend/rag/ingest.py`.

## Configuration

All settings live in `backend/.env`. Only the API key is required.

| Variable | Default | What it does |
|---|---|---|
| `OPENAI_API_KEY` | | OpenAI key (1st choice) |
| `ANTHROPIC_API_KEY` | | Anthropic key (2nd choice) |
| `GOOGLE_API_KEY` | | Google AI Studio key (3rd choice). `GEMINI_API_KEY` also works. |
| `LLM_PROVIDER` | unset | Put `openai`, `anthropic` or `google` first |
| `EMBED_PROVIDER` | auto | Force `openai`, `google` or `local` embeddings |
| `OPENAI_MODEL` | `openai/gpt-5.4` | Any [LiteLLM model string](https://docs.litellm.ai/docs/providers) |
| `ANTHROPIC_MODEL` | `anthropic/claude-sonnet-5-5` | Any LiteLLM model string |
| `GEMINI_MODEL` | `gemini/gemini-2.5-pro` | Any LiteLLM model string |
| `RAG_API_PORT` | `8710` | Backend port |
| `RAG_BROWSER` | auto-detected | Path to Chrome, Edge or Chromium for PDF printing |
| `RAG_DATA_DIR` | `backend/data` | Where your profile, index, caches and runs are stored |

## How it works

```
                      ┌────────────────────────────── once ──────────────────────────────┐
 resume.pdf / .json ─▶│ extract ─▶ normalize ─▶ chunk (parent/child) ─▶ embed ─▶ ChromaDB │
                      └──────────────────────────────────────────────────────────────────┘

                      ┌────────────────────────── per job post ──────────────────────────┐
 job post ───────────▶│ detect company/role ─▶ split into requirement lines              │
                      │   ─▶ hybrid retrieve (BM25 + vectors, RRF) per requirement        │
                      │   ─▶ plan: each requirement gets one home section                 │
                      │   ─▶ write section by section from evidence only                  │
                      │   ─▶ validate (keywords + numbers must be in the evidence)        │
                      │   ─▶ humanize ─▶ validate again ─▶ coverage report                │
                      │   ─▶ render HTML ─▶ one-page fit loop ─▶ resume.pdf               │
                      └──────────────────────────────────────────────────────────────────┘
```

**The validation rule.** A bullet passes only if:

- every job-post keyword it uses appears in the job post, the bullet **and** the cited evidence, and
- every number in it appears in the cited evidence.

Bullets that fail are rewritten or dropped. They are never shipped.

**Coverage labels**

| Label | Meaning |
|---|---|
| **PROVEN** | A validated bullet on the page backs this requirement |
| **MISSED** | Evidence exists in your profile, but it didn't make it onto the page |
| **GAP** | No evidence in your profile for this requirement |
| **ROUTED** | Handled outside the bullets (for example education or soft skills) |

- **Coverage** = PROVEN / (PROVEN + MISSED + GAP): how much of the job post the page proves.
- **Recall** = PROVEN / (PROVEN + MISSED): how much of your *available* evidence the agent used.

**Caching and cost.** LLM responses are cached on disk by exact prompt. Re-running the same post costs $0,
while a different post means new prompts. Semantic caching is deliberately not used, because it would hand
back bullets written for a different job.

## Command line

The agent can also run without the web app:

```bash
# tailor from a text file
python backend/rag/agent.py my_jd.txt --company "Halcyon Freight" --role "AI Engineer"

# show the evidence each field would use, without any LLM calls
python backend/rag/agent.py my_jd.txt --dry-run

# rebuild the vector index from backend/data/profile.json
python backend/rag/ingest.py

# chunk only and print stats, no API calls
python backend/rag/ingest.py --dry-run
```

<details>
<summary>HTTP API (backend on :8710)</summary>

| Method | Path | What |
|---|---|---|
| `GET` | `/api/health` | Profile and index status, active provider, missing keys |
| `POST` | `/api/source` | Upload a resume (`.pdf` or `.json`), then chunk and embed it |
| `GET` | `/api/profile` | The active profile |
| `GET` | `/api/profile/template` | An empty profile to fill in by hand |
| `POST` | `/api/detect` | `{jd, company?, role?}` → company and role, with options if unsure |
| `POST` | `/api/runs` | `{jd, company, role}` → start a run, returns `{job_id}` |
| `GET` | `/api/jobs/{id}` | Live status of a run |
| `GET` | `/api/runs` | All saved runs with coverage scores |
| `GET` | `/api/runs/{slug}` | Resume, coverage and render info for one run |
| `PUT` | `/api/runs/{slug}/resume` | Save edits |
| `POST` | `/api/runs/{slug}/render` | Rebuild the one-page PDF |
| `GET` | `/api/runs/{slug}/pdf` | The PDF |
| `GET` | `/api/costs` | LLM spend per job post and in total |
| `POST` | `/api/lint` | `{text}` → style and humanizer checks on one line |
| `POST` | `/api/ingest` | Re-chunk and re-embed the profile |

Only one run or ingest happens at a time.
</details>

## Project layout

```
backend/
  server.py              FastAPI: upload, runs, live job status, edits, PDF builder, costs
  .env.example           copy to .env and add one API key
  rag/
    providers.py         which LLM / embedding provider to use, fallback order
    writer.py            the only door to an LLM: LiteLLM, exact-match disk cache, cost trace
    profile_schema.py    profile shape, normalizer, PDF -> profile extraction
    ingest.py            profile -> parent/child chunks -> embeddings -> ChromaDB
    jd_meta.py           company + role from the job post, with confirm-if-unsure
    jd_split.py          job post -> tagged requirement lines
    retrieve.py          multi-query hybrid search (standalone explorer)
    agent.py             the LangGraph resume agent
    coverage.py          requirement coverage report
    render.py            resume.json -> HTML -> one-page fit loop
    pdf_builder.py       resume template CSS, browser printing
    aliases.json         generic tech synonyms (Postgres = PostgreSQL, ...)
  data/                  YOUR data (gitignored): profile, index, caches, runs
frontend/                Vite + React + TypeScript + Tailwind + Framer Motion + react-pdf
examples/                a fictional profile to try the app with
start.sh / start.ps1     start backend and frontend together
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Yellow banner: *"Add one of OPENAI_API_KEY…"* | Add a key to `backend/.env`, then **restart the backend**. |
| Banner still shows after adding a key | The old backend is still running. Close it and start it again. |
| *"The index was built with … embeddings"* | Your keys now select a different embedding model. Re-index (**Re-index** in the app). |
| *"Could not read text from this PDF"* | It's probably a scanned image. Upload a text-based PDF or a JSON profile. |
| PDF build fails | Install Chrome, Edge or Chromium, or set `RAG_BROWSER` to its path. |
| *"Another run is in progress"* | Only one job runs at a time. Wait for it to finish. |
| *"Every configured LLM provider failed"* | Each provider's error is listed. Check the key, your quota and the model name. |
| Port already in use | Set `RAG_API_PORT`, or stop whatever is using 8710 or 5180. |

## Privacy

- Everything runs locally. There is no server of ours and no telemetry.
- Your resume and job posts are sent **only** to the LLM and embedding providers whose keys you add. With an
  Anthropic-only setup, embeddings never leave your machine.
- `backend/.env` (your keys) and `backend/data/` (your profile, index and generated resumes) are gitignored.
  Don't commit them.

## Contributing

Issues and pull requests are welcome.

1. Fork the repo and create a branch.
2. Keep the core rule intact: **nothing reaches the resume without evidence.**
3. Check your change:
   ```bash
   cd frontend && npx tsc --noEmit             # types
   python backend/rag/ingest.py --dry-run      # chunking, no API calls
   python backend/rag/agent.py my_jd.txt --dry-run
   ```
4. Open a pull request describing what changed and why.

## License

[MIT](LICENSE). Use it, fork it, ship it.
