import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle, Circle, XCircle } from "@phosphor-icons/react";
import type { JobInfo } from "../api";
import { BreathingDot, Label, snappy } from "./ui";

const WHAT: Record<string, string> = {
  prepare: "Split the post, embed requirements, rank every chunk",
  plan: "Give each requirement one home section",
  experience: "Write job bullets from their own evidence",
  projects: "Pick projects by new coverage",
  research: "Fill both research slots",
  education: "Select degrees and thesis line",
  summary: "Summarize what the page already proves",
  humanize: "Rewrite AI tells, then validate again",
  skills: "Select JD-named skills",
  certifications: "Rank certifications",
  assemble: "Save resume.json, evidence map, coverage",
  render: "Fit one page and print the PDF",
  read_pdf: "Read the text out of your PDF",
  extract: "LLM copies the facts into a structured profile",
  validate: "Check the JSON has a name and something to prove",
  load_profile: "Read the profile JSON",
  chunk: "Structure-based parent-child chunks",
  merge_near_dups: "Merge near-duplicate chunks",
  store: "Embed and write to ChromaDB",
};

export default function Pipeline({ job }: { job: JobInfo }) {
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [job.log.length]);

  const done = job.steps.filter((s) => s.state === "done").length;
  const elapsed = Math.round(((job.finished ?? Date.now() / 1000) - job.started));

  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-3">
          <Label>{job.kind === "ingest" ? "Indexing your resume" : "Agent run"}</Label>
          <h1 className="text-3xl font-semibold tracking-tighter md:text-4xl">
            {job.status === "error" ? "The run stopped." : job.status === "done" ? "Done." : "Working through the graph."}
          </h1>
        </div>
        <div className="flex items-baseline gap-5 font-mono text-xs text-zinc-500">
          <span><span className="text-lg text-ink">{done}</span>/{job.steps.length} nodes</span>
          <span><span className="text-lg text-ink">{elapsed}</span>s</span>
        </div>
      </header>

      {job.error && (
        <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 font-mono text-xs text-rose-700">{job.error}</div>
      )}

      <div className="grid gap-8 md:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <ol className="relative flex flex-col">
          <span className="absolute bottom-3 left-[9px] top-3 w-px bg-zinc-200" />
          {job.steps.map((s) => (
            <motion.li key={s.name} layout transition={snappy} className="relative flex items-start gap-3 py-2">
              <span className="relative z-[1] mt-0.5 flex size-[19px] items-center justify-center rounded-full bg-canvas">
                {s.state === "done" ? <CheckCircle size={19} weight="fill" className="text-accent" />
                  : s.state === "error" ? <XCircle size={19} weight="fill" className="text-rose-500" />
                  : s.state === "running" ? <BreathingDot />
                  : <Circle size={15} className="text-zinc-300" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className={`flex items-baseline justify-between gap-3 font-mono text-[13px] ${s.state === "pending" ? "text-zinc-400" : "text-ink"}`}>
                  {s.name}
                  {s.ms != null && <span className="text-[11px] text-zinc-400">{(s.ms / 1000).toFixed(1)}s</span>}
                </span>
                <AnimatePresence initial={false}>
                  {(s.state === "running" || s.state === "error") && (
                    <motion.span initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                      className="block text-xs text-zinc-500">
                      {WHAT[s.name] ?? ""}
                    </motion.span>
                  )}
                </AnimatePresence>
              </span>
            </motion.li>
          ))}
        </ol>

        <div className="flex min-h-[320px] flex-col overflow-hidden rounded-2xl border border-night-fg/5 bg-night text-night-fg/85 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]">
          <div className="flex items-center justify-between border-b border-night-fg/5 px-4 py-2.5">
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-night-dim">stdout</span>
            {job.status === "running" && <BreathingDot />}
          </div>
          <div ref={logRef} className="scroll-thin max-h-[460px] flex-1 overflow-y-auto px-4 py-3 font-mono text-[11.5px] leading-[1.7]">
            {job.log.length === 0 && <span className="text-night-faint">waiting for the first node...</span>}
            {job.log.map((l, i) => (
              <div key={i} className={l.startsWith("COVERAGE") ? "text-emerald-300" : l.startsWith("NOTE") ? "text-amber-300/80" : ""}>
                {l}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
