import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Coins, Database, WarningCircle } from "@phosphor-icons/react";
import { api, ago, usd, type CostOverview, type Health, type JobInfo, type RunDetail, type RunSummary } from "./api";
import CostPanel from "./components/CostPanel";
import SourceView from "./components/SourceView";
import Sidebar from "./components/Sidebar";
import Compose from "./components/Compose";
import Pipeline from "./components/Pipeline";
import RunView from "./components/RunView";
import PdfPanel from "./components/PdfPanel";
import { BreathingDot, GhostButton } from "./components/ui";
import ThemeToggle from "./components/ThemeToggle";

type View = "boot" | "source" | "compose" | "job" | "run";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [view, setView] = useState<View>("boot");
  const [slug, setSlug] = useState<string | null>(null);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [runLoading, setRunLoading] = useState(false);
  const [job, setJob] = useState<JobInfo | null>(null);
  const [pdfVersion, setPdfVersion] = useState(() => Date.now());
  const [building, setBuilding] = useState(false);
  const [buildError, setBuildError] = useState<string | null>(null);
  const [costs, setCosts] = useState<CostOverview | null>(null);
  const [costOpen, setCostOpen] = useState(() => {
    try { return localStorage.getItem("proofline-cost-panel") === "open"; } catch { return false; }
  });
  const toggleCost = useCallback((next?: boolean) => {
    setCostOpen((prev) => {
      const v = next ?? !prev;
      try { localStorage.setItem("proofline-cost-panel", v ? "open" : "closed"); } catch { /* per-visit only */ }
      return v;
    });
  }, []);
  const closeCost = useCallback(() => toggleCost(false), [toggleCost]);

  const refresh = useCallback(async () => {
    try {
      const [h, r, c] = await Promise.all([api.health(), api.runs(), api.costs()]);
      setHealth(h); setRuns(r); setCosts(c); setOffline(false);
      return { h, r };
    } catch {
      setOffline(true); setRuns((prev) => prev ?? []);
      return null;
    }
  }, []);

  const openRun = useCallback(async (s: string) => {
    setSlug(s); setView("run"); setRunLoading(true); setBuildError(null);
    try { setRun(await api.run(s)); setPdfVersion(Date.now()); }
    catch (e) { setBuildError((e as Error).message); }
    finally { setRunLoading(false); }
  }, []);

  useEffect(() => {
    refresh().then((res) => {
      if (!res) { setView("source"); return; }
      if (!res.h.profile.exists) setView("source");           // first run: upload a resume before anything else
      else if (res.r.length) openRun(res.r[0].slug);
      else setView("compose");
    });
  }, [refresh, openRun]);

  // poll the running job; cleanup on every change
  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") return;
    const t = setInterval(async () => {
      try {
        const j = await api.job(job.id);
        setJob(j);
        if (j.status === "done") {
          await refresh();
          if (j.kind === "agent" && j.slug) setTimeout(() => openRun(j.slug!), 900);
          if (j.kind === "ingest") setTimeout(() => setView("source"), 900);
        }
      } catch { /* server blip: keep polling */ }
    }, 800);
    return () => clearInterval(t);
  }, [job, refresh, openRun]);

  async function startJob(id: string, kind: "agent" | "ingest" = "agent") {
    setView("job");
    setJob({ id, kind, status: "queued", steps: [], log: [], slug: null, error: null, started: Date.now() / 1000, finished: null, render: null });
    setJob(await api.job(id));
  }

  async function reindex() {
    if (!window.confirm("Re-chunk and re-embed the profile into ChromaDB? This calls your embedding provider (OpenAI or Gemini; local if you only have an Anthropic key).")) return;
    try { const { job_id } = await api.ingest(); await startJob(job_id, "ingest"); }
    catch (e) { setBuildError((e as Error).message); }
  }

  async function build() {
    if (!slug) return;
    setBuilding(true); setBuildError(null);
    try { await api.render(slug); await openRun(slug); await refresh(); }
    catch (e) { setBuildError((e as Error).message); }
    finally { setBuilding(false); }
  }

  const busy = !!health?.busy || job?.status === "running" || job?.status === "queued";

  return (
    <div className="grain min-h-[100dvh]">
      <header className="sticky top-0 z-30 border-b border-zinc-200/70 bg-canvas/85 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1600px] items-center justify-between gap-4 px-4 md:px-6">
          <div className="flex items-center gap-3">
            <span className="flex size-7 items-center justify-center rounded-lg bg-ink">
              <svg viewBox="0 0 32 32" className="size-4"><path d="M9 24V8h7a5 5 0 010 10H9" stroke="#34a37a" strokeWidth="3.2" fill="none" strokeLinecap="round" /></svg>
            </span>
            <span className="text-[15px] font-semibold tracking-tight">RAG Resume Builder</span>
            <span className="hidden font-mono text-[11px] text-zinc-400 lg:inline">evidence-backed tailoring</span>
          </div>
          <div className="flex items-center gap-3">
            {offline ? (
              <span className="flex items-center gap-1.5 text-xs text-rose-600"><WarningCircle size={14} />API offline on :8710</span>
            ) : health && (
              <span className="hidden items-center gap-2 font-mono text-[11px] text-zinc-500 md:flex">
                <BreathingDot tone={busy ? "amber" : "accent"} />
                {busy ? "agent running" : `${health.index.chunks} chunks indexed`}
                {health.index.built_at && <span className="text-zinc-400">· {ago(health.index.built_at)}</span>}
                <span className="rounded-full border border-zinc-200 px-2 py-0.5 text-zinc-500">{health.model}</span>
              </span>
            )}
            <button type="button" onClick={() => toggleCost()} aria-expanded={costOpen} aria-label="LLM spend panel"
              title={costOpen ? "Hide spend panel (Esc)" : "Show LLM spend"}
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition active:scale-[0.98]
                ${costOpen ? "border-ink bg-ink text-white" : "border-zinc-200 bg-white text-zinc-700 hover:border-zinc-300 hover:text-ink"}`}>
              <Coins size={14} />
              <span className="font-mono">{costs ? usd(costs.total) : "--"}</span>
            </button>
            <ThemeToggle />
            <GhostButton onClick={reindex} disabled={busy || offline || !health?.profile.exists} title="Re-chunk and re-embed the profile">
              <Database size={14} />Re-index
            </GhostButton>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1920px] gap-8 px-4 py-6 md:px-6 lg:grid-cols-[220px_minmax(0,1fr)_minmax(0,1.15fr)] lg:gap-8">
        <Sidebar runs={runs} selected={slug} composing={view === "compose"}
          source={health?.profile ?? null} sourceActive={view === "source"}
          onSource={() => setView("source")}
          onNew={() => setView(job && job.status !== "done" && job.status !== "error" ? "job"
            : health?.profile.exists ? "compose" : "source")}
          onSelect={openRun} />

        <div className="min-w-0 pb-16">
          {buildError && (
            <div className="mb-6 flex items-center gap-2 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              <WarningCircle size={16} />{buildError}
            </div>
          )}
          <AnimatePresence mode="wait">
            <motion.div key={view === "run" ? `run-${slug}` : view}
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}>
              {view === "boot" && <RunSkeleton />}
              {view === "source" && <SourceView hasProfile={!!health?.profile.exists} busy={busy}
                missingKeys={health?.missing_keys ?? []} onStarted={(id) => startJob(id, "ingest")} />}
              {view === "compose" && <Compose busy={busy} onStarted={(id) => startJob(id)} />}
              {view === "job" && job && <Pipeline job={job} />}
              {view === "run" && (runLoading && !run ? <RunSkeleton /> : run && <RunView run={run} onSaved={async (s) => { await openRun(s); await refresh(); }} />)}
            </motion.div>
          </AnimatePresence>
        </div>

        <PdfPanel slug={view === "compose" || view === "boot" || view === "source" ? null : view === "job" ? job?.slug ?? null : slug}
          hasPdf={view === "run" ? !!run?.has_pdf : job?.status === "done"}
          version={pdfVersion} info={view === "run" ? run?.render ?? null : job?.render ?? null}
          building={building || (view === "job" && job?.steps.find((s) => s.name === "render")?.state === "running")}
          onBuild={build} />
      </main>

      <CostPanel open={costOpen} onClose={closeCost} overview={costs}
        selected={view === "run" ? slug : null}
        selectedCost={view === "run" && run?.slug === slug ? run?.cost ?? null : null}
        onSelect={openRun} />
    </div>
  );
}

function RunSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      <div className="shimmer h-3 w-32 rounded" />
      <div className="shimmer h-9 w-2/3 rounded-lg" />
      <div className="shimmer h-px w-full" />
      {[0, 1, 2, 3].map((i) => <div key={i} className="shimmer h-16 rounded-xl" />)}
    </div>
  );
}
