import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BracketsCurly, DownloadSimple, FilePdf, UploadSimple, WarningCircle } from "@phosphor-icons/react";
import { ago, api, type ProfileInfo } from "../api";
import { Label, snappy } from "./ui";

type Props = {
  hasProfile: boolean;
  busy: boolean;
  missingKeys: string[];
  onStarted: (jobId: string) => void;
};

function download(name: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

/** Where the evidence comes from: upload a resume PDF (LLM-extracted) or a profile JSON, then review it. */
export default function SourceView({ hasProfile, busy, missingKeys, onStarted }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [info, setInfo] = useState<ProfileInfo | null>(null);
  const [loading, setLoading] = useState(hasProfile);

  useEffect(() => {
    if (!hasProfile) { setInfo(null); setLoading(false); return; }
    let live = true;
    setLoading(true);
    api.profile().then((p) => { if (live) setInfo(p); }).catch(() => {}).finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [hasProfile]);

  async function send(file: File | undefined) {
    if (!file) return;
    const ext = file.name.toLowerCase().split(".").pop();
    if (ext !== "pdf" && ext !== "json") { setError("Use a .pdf resume or a .json profile."); return; }
    if (file.size > 8 * 1024 * 1024) { setError("That file is over 8 MB."); return; }
    setSending(true); setError(null);
    try { onStarted((await api.uploadSource(file)).job_id); }
    catch (e) { setError((e as Error).message); }
    finally { setSending(false); if (input.current) input.current.value = ""; }
  }

  const blocked = busy || sending || missingKeys.length > 0;

  return (
    <div className="flex flex-col gap-10">
      <header className="flex flex-col gap-3">
        <Label>{hasProfile ? "Your source" : "Step 1 of 2"}</Label>
        <h1 className="text-3xl font-semibold leading-[1.05] tracking-tighter md:text-4xl">
          {hasProfile ? "Everything the agent can prove." : "Start with your resume."}
          <span className="block text-zinc-400">
            {hasProfile ? "Replace it any time." : "It becomes the evidence every bullet must cite."}
          </span>
        </h1>
      </header>

      {missingKeys.length > 0 && (
        <div className="flex items-start gap-2.5 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
          <WarningCircle size={18} className="mt-0.5 shrink-0" />
          <span>
            Add <span className="font-mono">{missingKeys.join(" and ")}</span> to <span className="font-mono">backend/.env</span>,
            then restart the backend. Copy <span className="font-mono">backend/.env.example</span> to start.
          </span>
        </div>
      )}

      <div className="grid gap-8 md:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div className="flex flex-col gap-3">
          <button
            type="button"
            disabled={blocked}
            onClick={() => input.current?.click()}
            onDragOver={(e) => { e.preventDefault(); if (!blocked) setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); if (!blocked) send(e.dataTransfer.files[0]); }}
            className={`group relative flex min-h-[260px] flex-col items-start justify-end gap-3 overflow-hidden rounded-[1.75rem] border-2 border-dashed p-7 text-left transition
              ${drag ? "border-accent bg-accent-soft" : "border-zinc-300 bg-white hover:border-zinc-400"}
              disabled:cursor-not-allowed disabled:opacity-60 active:scale-[0.995]`}
          >
            <motion.span animate={drag ? { y: -6, scale: 1.05 } : { y: 0, scale: 1 }} transition={snappy}
              className="mb-auto flex size-12 items-center justify-center rounded-2xl bg-ink text-white">
              <UploadSimple size={22} />
            </motion.span>
            <span className="text-lg font-semibold tracking-tight">
              {sending ? "Uploading" : drag ? "Drop it" : hasProfile ? "Upload a new version" : "Drop your resume here"}
            </span>
            <span className="max-w-[44ch] text-sm leading-relaxed text-zinc-500">
              A <b className="font-medium text-ink">PDF</b> is read and copied into a profile by the LLM (one call).
              A <b className="font-medium text-ink">JSON</b> profile is used as is. Either way it's chunked and embedded.
            </span>
          </button>
          <input ref={input} type="file" accept=".pdf,.json,application/pdf,application/json" className="hidden"
            onChange={(e) => send(e.target.files?.[0])} />
          <AnimatePresence>
            {error && (
              <motion.span initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className="flex items-center gap-1.5 text-sm text-rose-600"><WarningCircle size={15} />{error}</motion.span>
            )}
          </AnimatePresence>
        </div>

        <aside className="flex flex-col gap-6 border-t border-zinc-200 pt-6 text-sm leading-relaxed text-zinc-500 md:border-l md:border-t-0 md:pl-6 md:pt-0">
          <div className="flex flex-col gap-2">
            <Label>PDF or JSON?</Label>
            <p>PDF is the quick start. JSON gives the agent more to work with: add every project and detail,
              even the ones that never fit on a one-page resume. More evidence means better tailoring.</p>
          </div>
          <div className="flex flex-col gap-2">
            <Label>Write your own</Label>
            <p>Start from the empty template, or upload a PDF first and download the profile it makes.</p>
            <div className="flex flex-wrap gap-2 pt-1">
              <button type="button" onClick={async () => download("profile.template.json", await api.template())}
                className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition hover:text-ink active:scale-[0.98]">
                <BracketsCurly size={14} />Empty template
              </button>
              {info && (
                <button type="button" onClick={() => download("profile.json", info.profile)}
                  className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition hover:text-ink active:scale-[0.98]">
                  <DownloadSimple size={14} />Current profile.json
                </button>
              )}
            </div>
          </div>
          <div className="flex flex-col gap-2">
            <Label>Privacy</Label>
            <p>Your files stay in <span className="font-mono text-xs">backend/data/</span>. Text is sent only to the
              LLM and embedding APIs you configure.</p>
          </div>
        </aside>
      </div>

      {loading && <div className="shimmer h-40 rounded-2xl" />}
      {info && !loading && <ProfileReview info={info} />}
    </div>
  );
}

function ProfileReview({ info }: { info: ProfileInfo }) {
  const s = info.summary;
  const groups: [string, string[]][] = [
    ["Experience", s.jobs], ["Projects", s.projects], ["Research", s.research], ["Education", s.education],
  ];
  return (
    <section className="flex flex-col gap-6 border-t border-zinc-200 pt-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <Label>Indexed profile</Label>
          <h2 className="text-2xl font-semibold tracking-tight">{s.name}</h2>
          {s.email && <span className="font-mono text-xs text-zinc-500">{s.email}</span>}
        </div>
        {info.source && (
          <span className="flex items-center gap-1.5 font-mono text-[11px] text-zinc-500">
            {info.source.kind === "pdf" ? <FilePdf size={14} /> : <BracketsCurly size={14} />}
            {info.source.file} · {ago(info.source.uploaded_at)}
          </span>
        )}
      </div>
      <dl className="flex flex-wrap gap-x-8 gap-y-3 font-mono text-xs text-zinc-500">
        {Object.entries(s.counts).map(([k, v]) => (
          <div key={k} className="flex items-baseline gap-2">
            <dd className="text-xl text-ink">{v}</dd><dt>{k.replace("_", " ")}</dt>
          </div>
        ))}
      </dl>
      <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2">
        {groups.filter(([, items]) => items.length).map(([title, items]) => (
          <div key={title} className="flex flex-col gap-2">
            <Label>{title}</Label>
            <ul className="stagger flex flex-col gap-1.5 text-sm text-zinc-700">
              {items.map((x, i) => <li key={x + i} style={{ ["--i" as string]: i }} className="truncate">{x}</li>)}
            </ul>
          </div>
        ))}
      </div>
      {info.source?.kind === "pdf" && (
        <p className="text-xs leading-relaxed text-zinc-400">
          Extracted from a PDF by the LLM. Check names, dates and numbers above. If something is off, download the
          profile.json, fix it, and upload the JSON.
        </p>
      )}
    </section>
  );
}
