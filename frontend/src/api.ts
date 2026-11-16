export type Bullet = {
  text: string;
  requirements: number[];
  keywords: string[];
  evidence: string[];
  filler?: boolean;
  draft?: string;
};

type Block = { heading: string; pool: string; bullets: Bullet[] };
export type Job_ = Block & { company: string; title: string; location: string; dates: string };
export type Project = Block & { name: string; link?: string; tech?: string[] };
export type Research = Block & { title: string; institution: string; location: string; dates: string; advisor?: string };
export type Education = { school: string; location: string; degree: string; gpa?: string; dates: string; thesis?: string };
export type Cert = { name: string; issuer: string; date?: string };

export type Resume = {
  company: string;
  role: string;
  display_order: string[];
  summary: { text: string; keywords: string[]; bullet_refs: string[]; draft?: string } | null;
  skills: { label: string; items: string[]; jd_hits: number }[] | null;
  experience: Job_[] | null;
  projects: Project[] | null;
  research: Research[] | null;
  education: Education[] | null;
  certifications: Cert[] | null;
};

export type Status = "PROVEN" | "MISSED" | "GAP" | "ROUTED";
export type CoverageRow = {
  n: number; mode: string; weight: number; text: string; status: Status; best: number; route: string; proofs: string[];
};
export type Coverage = {
  scores: {
    coverage: number; recall: number; term_hit_rate: number; filler?: number;
    counts: Record<Status, number>;
    terms: { have: number; shown: number; missing: string[] };
  };
  rows: CoverageRow[];
};

export type RenderInfo = { pdf: string; level: number; fill: number | null; trimmed: string[]; built_at: number };

export type CallTrace = {
  tag: string; model: string; cached: boolean; cost: number; saved: number | null; ms: number;
  tokens: { prompt: number; completion: number; reasoning?: number; cached_input?: number } | null;
};
export type Build = {
  at: number; model: string; calls: number; cache_hits: number; cost: number; saved: number; saved_unknown: number;
  tokens: { prompt: number; completion: number; reasoning: number };
  stages: Record<string, { calls: number; cost: number; saved: number }>;
  trace: CallTrace[];
};
export type CostFile = { total: number; saved: number; builds: Build[] };
export type CostBrief = {
  total: number; saved: number; builds: number;
  last: { at: number; cost: number; saved: number; calls: number; cache_hits: number; model: string };
};
export type CostOverview = {
  model: string; total: number; saved: number; builds: number; untracked: number;
  runs: { slug: string; company: string; role: string; cost: CostBrief | null }[];
};

export type RunSummary = {
  slug: string; company: string; role: string; updated: number; has_pdf: boolean;
  scores: Coverage["scores"] | null; cost: CostBrief | null;
};

export type RunDetail = {
  slug: string; resume: Resume; coverage: Coverage | null; render: RenderInfo | null; has_pdf: boolean;
  cost: CostFile | null;
};

export type StepState = "pending" | "running" | "done" | "error";
export type JobInfo = {
  id: string; kind: "agent" | "ingest"; status: "queued" | "running" | "done" | "error";
  steps: { name: string; state: StepState; ms?: number }[];
  log: string[]; slug: string | null; error: string | null; started: number; finished: number | null;
  render: RenderInfo | null;
};

export type SourceMeta = { kind: "pdf" | "json"; file: string; uploaded_at: number };

export type Health = {
  ok: boolean; model: string; busy: boolean; missing_keys: string[];
  profile: { exists: boolean; name: string; jobs: number; projects: number; research: number; source: SourceMeta | null };
  index: { chunks: number; built_at: number | null };
};

export type ProfileSummary = {
  name: string; email: string;
  counts: Record<"work_experience" | "projects" | "research_experience" | "education" | "certifications" | "skills", number>;
  jobs: string[]; projects: string[]; research: string[]; education: string[];
};
export type ProfileInfo = { summary: ProfileSummary; source: SourceMeta | null; profile: Record<string, unknown> };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const isForm = init?.body instanceof FormData;
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: isForm ? init?.headers : { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => call<Health>("/health"),
  runs: () => call<RunSummary[]>("/runs"),
  costs: () => call<CostOverview>("/costs"),
  run: (slug: string) => call<RunDetail>(`/runs/${slug}`),
  start: (jd: string, company: string, role: string) =>
    call<{ job_id: string }>("/runs", { method: "POST", body: JSON.stringify({ jd, company, role }) }),
  job: (id: string) => call<JobInfo>(`/jobs/${id}`),
  save: (slug: string, resume: Resume) =>
    call<{ ok: boolean }>(`/runs/${slug}/resume`, { method: "PUT", body: JSON.stringify(resume) }),
  render: (slug: string) => call<RenderInfo>(`/runs/${slug}/render`, { method: "POST" }),
  lint: (text: string) => call<{ issues: string[] }>("/lint", { method: "POST", body: JSON.stringify({ text }) }),
  ingest: () => call<{ job_id: string }>("/ingest", { method: "POST" }),
  uploadSource: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return call<{ job_id: string }>("/source", { method: "POST", body: form });
  },
  profile: () => call<ProfileInfo>("/profile"),
  template: () => call<Record<string, unknown>>("/profile/template"),
  detect: (jd: string, company: string, role: string) =>
    call<Detected>("/detect", { method: "POST", body: JSON.stringify({ jd, company, role }) }),
  pdfUrl: (slug: string, v: number) => `/api/runs/${slug}/pdf?v=${v}`,
};

export const usd = (x: number | null | undefined) =>
  x == null ? "--" : x === 0 ? "$0.00" : x < 0.01 ? `$${x.toFixed(4)}` : `$${x.toFixed(3)}`;

export const pct = (x: number | undefined | null) => (x == null ? "--" : `${Math.round(x * 100)}%`);

export const ago = (t: number) => {
  const s = Math.max(1, Math.round(Date.now() / 1000 - t));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
};

export type Detected = {
  company: string; role: string;
  company_options: string[]; role_options: string[];
  needs_confirm: boolean; source: "given" | "llm" | "error"; error?: string;
};
