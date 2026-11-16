import { motion } from "framer-motion";
import { BracketsCurly, FilePdf, FilePlus, FileText } from "@phosphor-icons/react";
import { ago, pct, usd, type Health, type RunSummary } from "../api";
import { Label, spring } from "./ui";

type Props = {
  runs: RunSummary[] | null;
  selected: string | null;
  composing: boolean;
  onNew: () => void;
  onSelect: (slug: string) => void;
  source: Health["profile"] | null;
  sourceActive: boolean;
  onSource: () => void;
};

export default function Sidebar({ runs, selected, composing, onNew, onSelect, source, sourceActive, onSource }: Props) {
  return (
    <aside className="flex flex-col gap-6 lg:sticky lg:top-[72px] lg:h-[calc(100dvh-88px)]">
      <button type="button" onClick={onSource}
        className={`flex items-center gap-3 rounded-2xl border px-4 py-3 text-left transition active:scale-[0.98]
          ${sourceActive ? "border-accent/50 bg-accent-soft" : "border-zinc-200 bg-white hover:border-zinc-300"}`}>
        <span className={`flex size-8 shrink-0 items-center justify-center rounded-xl ${source?.exists ? "bg-ink text-white" : "bg-zinc-100 text-zinc-500"}`}>
          {source?.source?.kind === "json" ? <BracketsCurly size={16} /> : <FilePdf size={16} />}
        </span>
        <span className="min-w-0">
          <span className="block truncate text-sm font-medium">{source?.exists ? source.name : "Upload your resume"}</span>
          <span className="block truncate text-xs text-zinc-500">
            {source?.exists ? `${source.jobs} jobs · ${source.projects} projects · ${source.research} research` : "PDF or JSON, step 1"}
          </span>
        </span>
      </button>
      <button
        type="button"
        onClick={onNew}
        className={`group flex items-center justify-between rounded-2xl border px-4 py-3 text-left text-sm transition active:scale-[0.98]
          ${composing ? "border-ink bg-ink text-white" : "border-zinc-200 bg-white text-ink hover:border-zinc-300"}`}
      >
        <span className="font-medium">New tailor</span>
        <FilePlus size={18} weight="regular" className={composing ? "text-white/70" : "text-zinc-400 group-hover:text-ink"} />
      </button>

      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <div className="flex items-baseline justify-between px-1">
          <Label>Runs</Label>
          <span className="font-mono text-[10px] text-zinc-400">{runs?.length ?? 0}</span>
        </div>

        {runs === null && (
          <div className="flex flex-col gap-2">
            {[0, 1, 2].map((i) => <div key={i} className="shimmer h-[62px] rounded-xl" />)}
          </div>
        )}

        {runs?.length === 0 && (
          <div className="rounded-xl border border-dashed border-zinc-300 px-4 py-6 text-sm leading-relaxed text-zinc-500">
            No runs yet. Paste a job post under <span className="text-ink">New tailor</span> and the agent saves each
            tailored resume here.
          </div>
        )}

        <ul className="scroll-thin stagger -mx-1 flex min-h-0 flex-col gap-1 overflow-y-auto px-1">
          {runs?.map((r, i) => {
            const active = r.slug === selected && !composing;
            const cov = r.scores?.coverage ?? 0;
            return (
              <li key={r.slug} style={{ ["--i" as string]: i }}>
                <button
                  type="button"
                  onClick={() => onSelect(r.slug)}
                  className="relative w-full rounded-xl px-3 py-2.5 text-left transition active:scale-[0.99]"
                >
                  {active && (
                    <motion.span
                      layoutId="run-active"
                      transition={spring}
                      className="absolute inset-0 rounded-xl border border-zinc-200 bg-white shadow-[0_8px_20px_-14px_rgba(28,28,31,0.25)]"
                    />
                  )}
                  <span className="relative flex items-start gap-2.5">
                    <FileText size={16} className={`mt-0.5 shrink-0 ${active ? "text-accent" : "text-zinc-400"}`} />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium">{r.company || "Untitled"}</span>
                      <span className="block truncate text-xs text-zinc-500">{r.role || r.slug}</span>
                      <span className="mt-2 flex items-center gap-2">
                        <span className="h-[3px] flex-1 overflow-hidden rounded-full bg-zinc-200/80">
                          <span className="block h-full rounded-full bg-accent" style={{ width: `${cov * 100}%` }} />
                        </span>
                        <span className="font-mono text-[10px] text-zinc-500">{pct(r.scores?.coverage)}</span>
                        {r.cost && <span className="font-mono text-[10px] text-zinc-500" title="LLM spend for this job post">{usd(r.cost.total)}</span>}
                        <span className="font-mono text-[10px] text-zinc-400">{ago(r.updated)}</span>
                      </span>
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </aside>
  );
}
