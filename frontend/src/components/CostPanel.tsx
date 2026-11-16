import { useEffect, useMemo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowSquareOut, Lightning, X } from "@phosphor-icons/react";
import { ago, usd, type CostFile, type CostOverview } from "../api";
import { Label, snappy, spring } from "./ui";

type Props = {
  open: boolean;
  onClose: () => void;
  overview: CostOverview | null;
  selected: string | null;
  selectedCost: CostFile | null;
  onSelect: (slug: string) => void;
};

/** Right-side LLM spend panel. Non-modal: the resume stays usable while it's open. Esc closes it. */
export default function CostPanel({ open, onClose, overview, selected, selectedCost, onSelect }: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <motion.aside
          key="cost-panel"
          role="complementary"
          aria-label="LLM spend"
          initial={{ x: "105%" }} animate={{ x: 0 }} exit={{ x: "105%" }} transition={spring}
          className="fixed bottom-3 right-3 top-[64px] z-40 flex w-[min(400px,calc(100vw-24px))] flex-col overflow-hidden rounded-[1.75rem]
            border border-zinc-200 bg-white shadow-[0_30px_60px_-30px_rgba(0,0,0,0.35),inset_0_1px_0_rgba(255,255,255,0.04)]"
        >
          <header className="flex items-center justify-between border-b border-zinc-200 px-5 py-3.5">
            <div className="flex items-center gap-2">
              <Label>LLM spend</Label>
              {overview && <span className="rounded-full border border-zinc-200 px-2 py-0.5 font-mono text-[10px] text-zinc-500">{overview.model}</span>}
            </div>
            <button type="button" onClick={onClose} aria-label="Hide spend panel"
              className="flex size-7 items-center justify-center rounded-full text-zinc-400 transition hover:bg-zinc-100 hover:text-ink active:scale-[0.94]">
              <X size={14} />
            </button>
          </header>

          <div className="scroll-thin flex-1 overflow-y-auto">
            {!overview ? <PanelSkeleton /> : (
              <>
                <section className="flex flex-col gap-1 px-5 pb-5 pt-5">
                  <span className="font-mono text-[40px] font-medium leading-none tracking-tighter">{usd(overview.total)}</span>
                  <span className="text-xs text-zinc-500">
                    total across {overview.builds} build{overview.builds === 1 ? "" : "s"}
                    {overview.saved > 0 && <> · <span className="text-accent">{usd(overview.saved)} saved by cache</span></>}
                  </span>
                  {overview.untracked > 0 && (
                    <span className="mt-1 text-[11px] text-zinc-400">
                      {overview.untracked} run{overview.untracked === 1 ? " was" : "s were"} built before cost tracking and show no spend.
                    </span>
                  )}
                </section>

                <section className="border-t border-zinc-200 px-3 py-3">
                  <div className="px-2 pb-2"><Label>Per job post</Label></div>
                  {overview.runs.length === 0 && (
                    <p className="px-2 py-4 text-sm text-zinc-500">No resumes yet. Spend shows up here after the first tailor.</p>
                  )}
                  <ul className="stagger flex flex-col">
                    {overview.runs.map((r, i) => {
                      const on = r.slug === selected;
                      const share = overview.total > 0 && r.cost ? r.cost.total / overview.total : 0;
                      return (
                        <li key={r.slug} style={{ ["--i" as string]: i }}>
                          <button type="button" onClick={() => onSelect(r.slug)}
                            className="relative w-full rounded-xl px-2 py-2 text-left transition active:scale-[0.99]">
                            {on && <motion.span layoutId="cost-active" transition={snappy} className="absolute inset-0 rounded-xl bg-zinc-100" />}
                            <span className="relative flex items-baseline justify-between gap-3">
                              <span className="min-w-0">
                                <span className="block truncate text-sm font-medium">{r.company || r.slug}</span>
                                <span className="block truncate text-xs text-zinc-500">{r.role}</span>
                              </span>
                              <span className="shrink-0 text-right">
                                <span className="block font-mono text-sm">{r.cost ? usd(r.cost.total) : "--"}</span>
                                <span className="block font-mono text-[10px] text-zinc-400">
                                  {r.cost ? `${r.cost.builds} build${r.cost.builds === 1 ? "" : "s"} · ${ago(r.cost.last.at)}` : "untracked"}
                                </span>
                              </span>
                            </span>
                            <span className="relative mt-2 block h-[3px] overflow-hidden rounded-full bg-zinc-200/70">
                              <motion.span className="block h-full rounded-full bg-accent" initial={{ width: 0 }}
                                animate={{ width: `${share * 100}%` }} transition={spring} />
                            </span>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </section>

                <AnimatePresence mode="wait">
                  {selected && (
                    <motion.div key={selected} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}>
                      <RunBreakdown cost={selectedCost} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}

function RunBreakdown({ cost }: { cost: CostFile | null }) {
  const last = cost?.builds[cost.builds.length - 1];
  const stages = useMemo(() => {
    if (!last) return [];
    const rows = Object.entries(last.stages).map(([name, s]) => ({ name, ...s, weight: s.cost + s.saved }));
    const max = Math.max(...rows.map((r) => r.weight), 1e-9);
    return rows.sort((a, b) => b.weight - a.weight).map((r) => ({ ...r, share: r.weight / max }));
  }, [last]);

  if (!cost || !last) return (
    <section className="border-t border-zinc-200 px-5 py-5 text-sm leading-relaxed text-zinc-500">
      This resume was built before cost tracking. Re-run the job post to trace its spend.
    </section>
  );

  return (
    <section className="flex flex-col gap-5 border-t border-zinc-200 px-5 py-5">
      <div className="flex items-baseline justify-between">
        <Label>Last build</Label>
        <span className="font-mono text-[10px] text-zinc-400">{ago(last.at)}</span>
      </div>

      <dl className="grid grid-cols-[1fr_auto] gap-y-1.5 font-mono text-[12px]">
        <dt className="text-zinc-500">spent</dt><dd className="text-right">{usd(last.cost)}</dd>
        <dt className="text-zinc-500">saved by cache</dt>
        <dd className="text-right text-accent">{usd(last.saved)}{last.saved_unknown > 0 && <span className="text-zinc-400"> +{last.saved_unknown} old</span>}</dd>
        <dt className="text-zinc-500">calls</dt><dd className="text-right">{last.calls} <span className="text-zinc-400">({last.cache_hits} cached)</span></dd>
        <dt className="text-zinc-500">tokens in / out</dt>
        <dd className="text-right">{last.tokens.prompt.toLocaleString()} / {last.tokens.completion.toLocaleString()}</dd>
      </dl>

      <div className="flex flex-col gap-2.5">
        <Label>By stage</Label>
        {stages.map((s) => (
          <div key={s.name} className="grid grid-cols-[88px_minmax(0,1fr)_64px] items-center gap-3 text-xs">
            <span className="truncate font-mono text-zinc-600">{s.name}</span>
            <span className="h-1.5 overflow-hidden rounded-full bg-zinc-100">
              <motion.span className={`block h-full rounded-full ${s.cost > 0 ? "bg-ink" : "bg-accent/60"}`}
                initial={{ width: 0 }} animate={{ width: `${Math.max(s.share * 100, 3)}%` }} transition={spring} />
            </span>
            <span className="text-right font-mono text-zinc-600">{s.cost > 0 ? usd(s.cost) : <span className="text-accent">cached</span>}</span>
          </div>
        ))}
      </div>

      <details className="group">
        <summary className="flex cursor-pointer list-none items-center justify-between text-xs text-zinc-500 hover:text-ink">
          <span>Every call ({last.trace.length})</span>
          <ArrowSquareOut size={12} className="transition group-open:rotate-90" />
        </summary>
        <ul className="mt-3 divide-y divide-zinc-200/70">
          {last.trace.map((c, i) => (
            <li key={i} className="flex items-baseline justify-between gap-3 py-2 text-[11px]">
              <span className="min-w-0 truncate text-zinc-600" title={c.tag}>{c.tag || "call"}</span>
              <span className="flex shrink-0 items-baseline gap-2 font-mono">
                {c.tokens && !c.cached && <span className="text-zinc-400">{c.tokens.prompt + c.tokens.completion}t</span>}
                {c.cached ? <span className="flex items-center gap-0.5 text-accent"><Lightning size={10} weight="fill" />cache</span>
                  : <span>{usd(c.cost)}</span>}
              </span>
            </li>
          ))}
        </ul>
      </details>

      {cost.builds.length > 1 && (
        <div className="flex flex-col gap-2">
          <Label>Build history</Label>
          <ul className="flex flex-col gap-1 font-mono text-[11px]">
            {[...cost.builds].reverse().map((b) => (
              <li key={b.at} className="flex justify-between text-zinc-500">
                <span>{ago(b.at)}</span>
                <span className="text-ink">{usd(b.cost)} <span className="text-zinc-400">· {b.cache_hits}/{b.calls} cached</span></span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function PanelSkeleton() {
  return (
    <div className="flex flex-col gap-4 p-5">
      <div className="shimmer h-10 w-32 rounded-lg" />
      <div className="shimmer h-3 w-48 rounded" />
      {[0, 1, 2].map((i) => <div key={i} className="shimmer h-12 rounded-xl" />)}
    </div>
  );
}
