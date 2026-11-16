import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowsClockwise, CaretDown, PencilSimple, Sparkle, WarningCircle } from "@phosphor-icons/react";
import { api, pct, type Bullet, type CoverageRow, type Resume, type RunDetail } from "../api";
import { Label, MagneticButton, StatusPill, snappy, spring } from "./ui";

type Tab = "resume" | "coverage";

export default function RunView({ run, onSaved }: { run: RunDetail; onSaved: (slug: string) => Promise<void> }) {
  const [tab, setTab] = useState<Tab>("resume");
  const [draft, setDraft] = useState<Resume>(run.resume);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { setDraft(run.resume); setDirty(false); setError(null); }, [run]);

  const s = run.coverage?.scores;
  const edit = (fn: (r: Resume) => void) => {
    setDraft((prev) => { const next = structuredClone(prev); fn(next); return next; });
    setDirty(true);
  };

  async function saveAndBuild() {
    setSaving(true);
    setError(null);
    try {
      await api.save(run.slug, draft);
      await api.render(run.slug);
      await onSaved(run.slug);
      setDirty(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <header className="grid gap-6 md:grid-cols-[minmax(0,1fr)_auto] md:items-end">
        <div className="flex flex-col gap-3">
          <Label>{run.slug}</Label>
          <h1 className="text-3xl font-semibold leading-[1.05] tracking-tighter md:text-4xl">
            {run.resume.company || "Untitled"}
            <span className="block text-zinc-400">{run.resume.role}</span>
          </h1>
        </div>
        {s && (
          <div className="flex items-end gap-6">
            <div className="flex flex-col">
              <span className="font-mono text-5xl font-medium leading-none tracking-tighter text-accent">{pct(s.coverage)}</span>
              <span className="mt-1 text-xs text-zinc-500">coverage</span>
            </div>
            <dl className="grid grid-cols-2 gap-x-5 gap-y-1 border-l border-zinc-200 pl-5 font-mono text-[11px] text-zinc-500">
              <dt>recall</dt><dd className="text-right text-ink">{pct(s.recall)}</dd>
              <dt>terms</dt><dd className="text-right text-ink">{s.terms.shown}/{s.terms.have}</dd>
              <dt>missed</dt><dd className="text-right text-ink">{s.counts.MISSED}</dd>
              <dt>gaps</dt><dd className="text-right text-ink">{s.counts.GAP}</dd>
            </dl>
          </div>
        )}
      </header>

      <nav className="flex gap-1 border-b border-zinc-200">
        {(["resume", "coverage"] as Tab[]).map((t) => (
          <button key={t} type="button" onClick={() => setTab(t)}
            className={`relative px-3 pb-2.5 pt-1 text-sm capitalize transition ${tab === t ? "text-ink" : "text-zinc-400 hover:text-zinc-600"}`}>
            {t === "coverage" ? `Evidence map${run.coverage ? ` (${run.coverage.rows.length})` : ""}` : "Resume"}
            {tab === t && <motion.span layoutId="tab-line" transition={spring} className="absolute inset-x-2 -bottom-px h-[2px] rounded-full bg-ink" />}
          </button>
        ))}
      </nav>

      <AnimatePresence mode="wait">
        <motion.div key={tab} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}>
          {tab === "resume" ? <ResumeEditor resume={draft} edit={edit} /> : <EvidenceMap rows={run.coverage?.rows ?? []} />}
        </motion.div>
      </AnimatePresence>

      <AnimatePresence>
        {(dirty || error) && (
          <motion.div initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 40, opacity: 0 }} transition={snappy}
            className="sticky bottom-4 z-20 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-night/95 px-4 py-3 text-sm text-night-fg backdrop-blur shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_20px_40px_-20px_rgba(28,28,31,0.6)]">
            <span className="flex items-center gap-2">
              {error ? <><WarningCircle size={16} className="text-rose-400" />{error}</> : "Unsaved edits on this resume."}
            </span>
            <div className="flex gap-2">
              <button type="button" onClick={() => { setDraft(run.resume); setDirty(false); setError(null); }}
                className="rounded-full px-3 py-1.5 text-xs text-night-dim hover:text-night-fg">Discard</button>
              <MagneticButton onClick={saveAndBuild} disabled={saving} className="!bg-night-fg !text-night !py-1.5 text-xs">
                <ArrowsClockwise size={14} className={saving ? "animate-spin" : ""} />
                {saving ? "Building PDF" : "Save and rebuild PDF"}
              </MagneticButton>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ---------- resume editor ---------- */

function ResumeEditor({ resume, edit }: { resume: Resume; edit: (fn: (r: Resume) => void) => void }) {
  return (
    <div className="flex flex-col gap-10">
      {resume.display_order.map((sec) => {
        if (sec === "summary" && resume.summary?.text) return (
          <Section key={sec} title="Summary">
            <EditableLine text={resume.summary.text} draft={resume.summary.draft} multiline
              onCommit={(t) => edit((r) => { r.summary!.text = t; })} />
          </Section>
        );
        if (sec === "skills" && resume.skills?.length) return (
          <Section key={sec} title="Skills">
            <dl className="grid gap-y-2 text-sm sm:grid-cols-[140px_minmax(0,1fr)]">
              {resume.skills.map((r) => (
                <div key={r.label} className="contents">
                  <dt className="font-mono text-[11px] uppercase tracking-wider text-zinc-400 sm:pt-0.5">{r.label}</dt>
                  <dd className="text-zinc-700">{r.items.join(", ")}</dd>
                </div>
              ))}
            </dl>
          </Section>
        );
        if (sec === "experience" && resume.experience) return (
          <Section key={sec} title="Experience">
            {resume.experience.map((j, ji) => (
              <Entry key={j.pool} title={j.company} sub={`${j.title}${j.location ? ` · ${j.location}` : ""}`} meta={j.dates}>
                <Bullets bullets={j.bullets} onCommit={(bi, t) => edit((r) => { r.experience![ji].bullets[bi].text = t; })} />
              </Entry>
            ))}
          </Section>
        );
        if (sec === "projects" && resume.projects?.length) return (
          <Section key={sec} title="Projects">
            {resume.projects.map((p, pi) => (
              <Entry key={p.pool} title={p.name} sub={(p.tech ?? []).slice(0, 7).join(" · ")} meta={p.link?.replace(/^https?:\/\//, "")}>
                <Bullets bullets={p.bullets} onCommit={(bi, t) => edit((r) => { r.projects![pi].bullets[bi].text = t; })} />
              </Entry>
            ))}
          </Section>
        );
        if (sec === "research" && resume.research?.length) return (
          <Section key={sec} title="Research">
            {resume.research.map((x, xi) => (
              <Entry key={x.pool} title={x.title.replace(/\s*[—–]\s*/, ": ")} sub={`${x.institution}${x.advisor ? ` · Advisor: ${x.advisor}` : ""}`} meta={x.dates}>
                <Bullets bullets={x.bullets} onCommit={(bi, t) => edit((r) => { r.research![xi].bullets[bi].text = t; })} />
              </Entry>
            ))}
          </Section>
        );
        if (sec === "education" && resume.education?.length) return (
          <Section key={sec} title="Education">
            {resume.education.map((e) => (
              <Entry key={e.school} title={e.school} sub={`${e.degree}${e.gpa ? ` · GPA ${e.gpa}` : ""}`} meta={e.dates}>
                {e.thesis && <p className="text-sm text-zinc-500">{e.thesis}</p>}
              </Entry>
            ))}
          </Section>
        );
        if (sec === "certifications" && resume.certifications?.length) return (
          <Section key={sec} title="Certifications">
            <ul className="flex flex-col gap-1 text-sm text-zinc-700">
              {resume.certifications.map((c) => (
                <li key={c.name}>{c.name} <span className="text-zinc-400">· {c.issuer}{c.date ? ` · ${c.date}` : ""}</span></li>
              ))}
            </ul>
          </Section>
        );
        return null;
      })}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="grid gap-4 md:grid-cols-[120px_minmax(0,1fr)]">
      <h2 className="pt-1 font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-400">{title}</h2>
      <div className="flex flex-col gap-6">{children}</div>
    </section>
  );
}

function Entry({ title, sub, meta, children }: { title: string; sub?: string; meta?: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 border-t border-zinc-200/80 pt-4 first:border-t-0 first:pt-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4">
        <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>
        {meta && <span className="font-mono text-[11px] text-zinc-400">{meta}</span>}
      </div>
      {sub && <p className="-mt-1 text-[13px] text-zinc-500">{sub}</p>}
      {children}
    </div>
  );
}

function Bullets({ bullets, onCommit }: { bullets: Bullet[]; onCommit: (i: number, text: string) => void }) {
  if (!bullets.length) return <p className="text-sm italic text-zinc-400">No bullets: nothing in this entry matched the post.</p>;
  return (
    <ul className="flex flex-col gap-3">
      {bullets.map((b, i) => (
        <li key={i} className="group grid grid-cols-[12px_minmax(0,1fr)] gap-2">
          <span className="mt-[9px] size-[5px] rounded-full bg-zinc-300 group-hover:bg-accent" />
          <div className="flex flex-col gap-1.5">
            <EditableLine text={b.text} draft={b.draft} onCommit={(t) => onCommit(i, t)} />
            <div className="flex flex-wrap items-center gap-1.5">
              {b.filler ? (
                <span className="rounded-md bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500">filler, proves nothing</span>
              ) : (
                b.requirements.map((r) => (
                  <span key={r} className="rounded-md bg-accent-soft px-1.5 py-0.5 font-mono text-[10px] text-accent">R{r}</span>
                ))
              )}
              {b.keywords.map((k) => (
                <span key={k} className="rounded-md border border-zinc-200 px-1.5 py-0.5 text-[11px] text-zinc-600">{k}</span>
              ))}
              <span className="ml-auto hidden font-mono text-[10px] text-zinc-400 group-hover:inline">
                {b.evidence.map((e) => e.split(".").slice(-1)[0]).join(", ")}
              </span>
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}

function EditableLine({ text, draft, onCommit, multiline }:
  { text: string; draft?: string; onCommit: (t: string) => void; multiline?: boolean }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(text);
  const [issues, setIssues] = useState<string[]>([]);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { setValue(text); }, [text]);
  useEffect(() => {
    if (editing && ref.current) {
      ref.current.focus();
      ref.current.style.height = `${ref.current.scrollHeight}px`;
    }
  }, [editing]);

  async function commit() {
    setEditing(false);
    const t = value.trim();
    if (!t || t === text) { setValue(text); return; }
    onCommit(t);
    try { setIssues((await api.lint(t)).issues); } catch { setIssues([]); }
  }

  return (
    <div className="flex flex-col gap-1">
      {editing ? (
        <textarea ref={ref} value={value} rows={multiline ? 4 : 2}
          onChange={(e) => { setValue(e.target.value); e.target.style.height = `${e.target.scrollHeight}px`; }}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Escape") { setValue(text); setEditing(false); }
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commit(); }
          }}
          className="w-full resize-none rounded-lg border border-accent/40 bg-white px-2.5 py-1.5 text-[14px] leading-relaxed outline-none ring-4 ring-accent/10" />
      ) : (
        <button type="button" onClick={() => setEditing(true)}
          className="group/line -mx-2.5 rounded-lg px-2.5 py-1 text-left text-[14px] leading-relaxed text-zinc-700 transition hover:bg-white">
          {text}
          <PencilSimple size={13} className="ml-1.5 inline text-zinc-300 opacity-0 transition group-hover/line:opacity-100" />
          {draft && draft !== text && (
            <span title={`Before the humanizer: ${draft}`} className="ml-1.5 inline-flex translate-y-[1px] items-center gap-0.5 font-mono text-[10px] text-zinc-400">
              <Sparkle size={11} />humanized
            </span>
          )}
        </button>
      )}
      {issues.length > 0 && (
        <span className="flex flex-wrap gap-x-3 text-xs text-amber-700">
          {issues.map((x) => <span key={x} className="flex items-center gap-1"><WarningCircle size={12} />{x}</span>)}
        </span>
      )}
    </div>
  );
}

/* ---------- evidence map ---------- */

function EvidenceMap({ rows }: { rows: CoverageRow[] }) {
  if (!rows.length) return (
    <p className="rounded-2xl border border-dashed border-zinc-300 px-5 py-8 text-sm text-zinc-500">
      No coverage file for this run. Re-run the agent to score it.
    </p>
  );
  return <ul className="stagger divide-y divide-zinc-200/80">{rows.map((r, i) => <Row key={r.n} row={r} i={i} />)}</ul>;
}

function Row({ row, i }: { row: CoverageRow; i: number }) {
  const [open, setOpen] = useState(false);
  return (
    <li style={{ ["--i" as string]: i }} className="py-3">
      <button type="button" onClick={() => setOpen((o) => !o)} disabled={!row.proofs.length}
        className="grid w-full grid-cols-[38px_minmax(0,1fr)_auto] items-start gap-3 text-left">
        <span className="pt-0.5 font-mono text-xs text-zinc-400">R{row.n}</span>
        <span className="flex flex-col gap-1.5">
          <span className="text-sm leading-relaxed text-zinc-700">{row.text}</span>
          <span className="flex items-center gap-2 font-mono text-[10px] text-zinc-400">
            <StatusPill status={row.status} />
            <span>{row.mode === "other" ? "duty" : row.mode}</span>
            <span>d={row.best.toFixed(3)}</span>
            {row.route !== "bullets" && <span>routed to {row.route}</span>}
          </span>
        </span>
        {row.proofs.length > 0 && (
          <CaretDown size={14} className={`mt-1 text-zinc-400 transition ${open ? "rotate-180" : ""}`} />
        )}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.ul initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }} className="ml-[50px] overflow-hidden">
            {row.proofs.map((p) => {
              const [where, ...rest] = p.split(": ");
              return (
                <li key={p} className="mt-2 border-l-2 border-accent/30 pl-3 text-[13px] leading-relaxed text-zinc-600">
                  <span className="block font-mono text-[10px] text-zinc-400">{where}</span>{rest.join(": ")}
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </li>
  );
}

