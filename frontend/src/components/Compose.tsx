import { useState } from "react";
import { ArrowRight, Check, Sparkle, WarningCircle } from "@phosphor-icons/react";
import { AnimatePresence, motion } from "framer-motion";
import { api } from "../api";
import { Label, MagneticButton } from "./ui";

const MIN = 200;

export default function Compose({ busy, onStarted }: { busy: boolean; onStarted: (jobId: string) => void }) {
  const [jd, setJd] = useState("");
  const [company, setCompany] = useState("");
  const [role, setRole] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [phase, setPhase] = useState<"idle" | "detecting" | "starting">("idle");
  // fields the LLM was unsure about -> chips to pick from (absent = nothing to confirm)
  const [confirm, setConfirm] = useState<{ company?: string[]; role?: string[] } | null>(null);
  const [detected, setDetected] = useState<string | null>(null);
  const short = jd.trim().length > 0 && jd.trim().length < MIN;

  async function start(c: string, r: string) {
    setPhase("starting");
    const { job_id } = await api.start(jd, c.trim(), r.trim());
    onStarted(job_id);
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (jd.trim().length < MIN) {
      setError(`Paste the full job post. ${MIN - jd.trim().length} more characters needed.`);
      return;
    }
    setSending(true);
    setError(null);
    try {
      // already confirming, or both names typed in: no LLM call
      if (confirm || (company.trim() && role.trim())) {
        await start(company, role);
        return;
      }
      setPhase("detecting");
      const d = await api.detect(jd, company, role);
      setCompany(d.company);
      setRole(d.role);
      if (!d.needs_confirm) {
        setDetected([d.role, d.company].filter(Boolean).join(" at "));
        await start(d.company, d.role);
        return;
      }
      setConfirm({
        ...(!company.trim() && !d.company ? { company: d.company_options } : {}),
        ...(!role.trim() && !d.role ? { role: d.role_options } : {}),
      });
      if (d.source === "error") setError("Could not read the names from the post. Type them in, or leave them blank.");
      setPhase("idle");
    } catch (err) {
      setError((err as Error).message);
      setPhase("idle");
    } finally {
      setSending(false);
    }
  }

  const resetDetection = () => { setConfirm(null); setDetected(null); };

  return (
    <form onSubmit={submit} className="grid gap-10 md:grid-cols-[minmax(0,1fr)_220px]">
      <div className="flex flex-col gap-8">
        <header className="flex flex-col gap-3">
          <Label>New tailor</Label>
          <h1 className="text-3xl font-semibold leading-[1.05] tracking-tighter md:text-4xl">
            Paste a job post.<br />
            <span className="text-zinc-400">Every bullet comes back with its proof.</span>
          </h1>
        </header>

        <div className="grid gap-5 sm:grid-cols-2">
          <Field id="company" label="Company" optional helper="Leave blank to read it from the post.">
            <input id="company" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="Halcyon Freight"
              className="input" />
          </Field>
          <Field id="role" label="Role" optional helper="Leave blank to read it from the post.">
            <input id="role" value={role} onChange={(e) => setRole(e.target.value)} placeholder="AI Engineer"
              className="input" />
          </Field>
        </div>

        <AnimatePresence>
          {confirm && (
            <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
              className="flex flex-col gap-4 rounded-2xl border border-amber-200 bg-amber-50/60 px-4 py-4 text-sm">
              <div className="flex items-start gap-2 text-amber-700">
                <Sparkle size={16} className="mt-0.5 shrink-0" />
                <span>The post does not state this clearly. Pick one, type your own above, or leave it blank.</span>
              </div>
              {confirm.company && <ChipRow label="Company" options={confirm.company} value={company} onPick={setCompany} />}
              {confirm.role && <ChipRow label="Role" options={confirm.role} value={role} onPick={setRole} />}
            </motion.div>
          )}
        </AnimatePresence>

        <Field id="jd" label="Job description" helper="Include the qualifications and responsibilities sections."
          error={error ?? (short ? `${MIN - jd.trim().length} more characters to go` : null)}>
          <textarea id="jd" value={jd} onChange={(e) => { setJd(e.target.value); setError(null); resetDetection(); }}
            rows={16} placeholder="About the job&#10;&#10;Qualifications&#10;Strong foundation in Computer Science..."
            className="input scroll-thin min-h-[320px] resize-y font-mono text-[13px] leading-relaxed" />
        </Field>

        <div className="flex items-center gap-4">
          <MagneticButton type="submit" disabled={busy || sending}>
            {phase === "detecting" ? "Reading the post" : phase === "starting" || sending ? "Starting"
              : busy ? "Agent busy" : confirm ? "Confirm and tailor" : "Tailor resume"}
            <ArrowRight size={16} weight="bold" />
          </MagneticButton>
          <span className="font-mono text-[11px] text-zinc-400">
            {detected ? <span className="inline-flex items-center gap-1"><Check size={12} />{detected}</span>
              : `${jd.trim().length.toLocaleString()} chars`}
          </span>
        </div>
      </div>

      <aside className="flex flex-col gap-6 border-t border-zinc-200 pt-6 text-sm leading-relaxed text-zinc-500 md:mt-24 md:border-l md:border-t-0 md:pl-6 md:pt-0">
        <div className="flex flex-col gap-2">
          <Label>What happens</Label>
          <p>The post is split into requirement lines. Each one searches your profile in ChromaDB (meaning plus keywords).</p>
        </div>
        <div className="flex flex-col gap-2">
          <Label>Then</Label>
          <p>A planner gives each requirement one home section. The writer only uses that evidence, and Python checks every keyword and number.</p>
        </div>
        <div className="flex flex-col gap-2">
          <Label>Last</Label>
          <p>A humanizer pass, a second validation, and a one-page PDF you can edit and rebuild.</p>
        </div>
      </aside>
    </form>
  );
}

function ChipRow({ label, options, value, onPick }:
  { label: string; options: string[]; value: string; onPick: (v: string) => void }) {
  const chips = [...options.map((o) => ({ text: o, value: o })), { text: "Not stated", value: "" }];
  return (
    <div className="flex flex-col gap-2">
      <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-500">{label}</span>
      <div className="flex flex-wrap gap-2" role="radiogroup" aria-label={label}>
        {chips.map((c) => {
          const on = value.trim().toLowerCase() === c.value.toLowerCase();
          return (
            <button key={c.text} type="button" role="radio" aria-checked={on} onClick={() => onPick(c.value)}
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition active:scale-[0.98] ${
                on ? "border-ink bg-ink text-white" : c.value ? "border-zinc-300 bg-white/70 text-ink hover:border-zinc-500"
                  : "border-dashed border-zinc-300 text-zinc-500 hover:border-zinc-500"}`}>
              {on && <Check size={12} weight="bold" />}{c.text}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Field({ id, label, helper, error, optional, children }:
  { id: string; label: string; helper?: string; error?: string | null; optional?: boolean; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-sm font-medium text-ink">
        {label}{optional && <span className="ml-1.5 font-normal text-zinc-400">optional</span>}
      </label>
      {children}
      {error ? (
        <span className="flex items-center gap-1.5 text-xs text-rose-600"><WarningCircle size={14} />{error}</span>
      ) : helper ? (
        <span className="text-xs text-zinc-400">{helper}</span>
      ) : null}
    </div>
  );
}
