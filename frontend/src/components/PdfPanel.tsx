import { useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import { ArrowsClockwise, ArrowsIn, ArrowsOut, DownloadSimple, FilePdf, MagnifyingGlassMinus, MagnifyingGlassPlus } from "@phosphor-icons/react";
import { api, ago, type RenderInfo } from "../api";
import { GhostButton, Label } from "./ui";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

type Props = {
  slug: string | null;
  hasPdf: boolean;
  version: number;
  info: RenderInfo | null;
  building: boolean;
  onBuild: () => void;
};

export default function PdfPanel({ slug, hasPdf, version, info, building, onBuild }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(520);
  const [zoom, setZoom] = useState(1);
  const [pages, setPages] = useState(0);
  const [failed, setFailed] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ x: number; y: number; left: number; top: number; id: number } | null>(null);
  const zoomBy = (d: number) => setZoom((z) => Math.min(3, Math.max(0.5, +(z + d).toFixed(2))));
  const canPan = zoom > 1;

  // Esc leaves full screen
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setExpanded(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  // Ctrl/Cmd + wheel zooms the PDF instead of the page (needs a non-passive listener)
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      zoomBy(e.deltaY < 0 ? 0.1 : -0.1);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // Click-and-drag to pan when zoomed in
  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = wrap.current;
    if (!canPan || !el || e.button !== 0 || (e.target as HTMLElement).closest("a,button")) return;
    drag.current = { x: e.clientX, y: e.clientY, left: el.scrollLeft, top: el.scrollTop, id: e.pointerId };
    el.setPointerCapture(e.pointerId);
    setDragging(true);
    e.preventDefault();
  };
  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = wrap.current, d = drag.current;
    if (!el || !d || d.id !== e.pointerId) return;
    el.scrollLeft = d.left - (e.clientX - d.x);
    el.scrollTop = d.top - (e.clientY - d.y);
  };
  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    wrap.current?.releasePointerCapture(drag.current.id);
    drag.current = null;
    setDragging(false);
    void e;
  };
  // Leaving zoom while dragging must not leave the cursor stuck
  useEffect(() => { if (!canPan) setDragging(false); }, [canPan]);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, e.contentRect.width - 32)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => { setFailed(null); }, [slug, version]);

  const url = slug && hasPdf ? api.pdfUrl(slug, version) : null;

  return (
    <section className={expanded
      ? "fixed inset-0 z-50 flex flex-col gap-3 bg-canvas/95 p-4 backdrop-blur-sm md:p-6"
      : "flex flex-col gap-3 lg:sticky lg:top-[72px] lg:h-[calc(100dvh-88px)]"}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-baseline gap-3">
          <Label>PDF</Label>
          {info && (
            <span className="font-mono text-[10px] text-zinc-400">
              {pages || 1}p · level {info.level} · fill {info.fill ? Math.round(info.fill * 100) : "--"}% · {ago(info.built_at)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <GhostButton title="Zoom out (Ctrl + scroll)" onClick={() => zoomBy(-0.15)} disabled={!url}>
            <MagnifyingGlassMinus size={14} />
          </GhostButton>
          <button type="button" title="Reset to fit width" onClick={() => setZoom(1)} disabled={!url}
            className="w-10 rounded text-center font-mono text-[10px] text-zinc-500 transition hover:text-zinc-300 disabled:opacity-50">
            {Math.round(zoom * 100)}%
          </button>
          <GhostButton title="Zoom in (Ctrl + scroll)" onClick={() => zoomBy(0.15)} disabled={!url}>
            <MagnifyingGlassPlus size={14} />
          </GhostButton>
          <GhostButton title={expanded ? "Exit full screen (Esc)" : "Full screen"} onClick={() => setExpanded((x) => !x)}>
            {expanded ? <ArrowsIn size={14} /> : <ArrowsOut size={14} />}
          </GhostButton>
          <GhostButton title="Build the PDF again" onClick={onBuild} disabled={!slug || building}>
            <ArrowsClockwise size={14} className={building ? "animate-spin" : ""} />{building ? "Building" : "Build"}
          </GhostButton>
          {url && (
            <a href={url} download={`Resume ${slug}.pdf`}
              className="inline-flex items-center gap-1.5 rounded-full bg-ink px-3 py-1.5 text-xs font-medium text-white transition hover:bg-zinc-800 active:scale-[0.98]">
              <DownloadSimple size={14} />PDF
            </a>
          )}
        </div>
      </div>

      <div ref={wrap}
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag} onPointerCancel={endDrag}
        className={`scroll-thin relative min-h-[520px] flex-1 overflow-auto rounded-[1.75rem] border border-zinc-200/70 bg-well p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.6)] dark:shadow-none ${canPan && url ? (dragging ? "cursor-grabbing select-none" : "cursor-grab select-none") : ""}`}
      >
        {!slug && <Empty title="No resume selected" body="Pick a run on the left, or start a new tailor. The one-page PDF shows up here." />}
        {slug && !hasPdf && !building && (
          <Empty title="No PDF built yet" body="Build runs the one-page fit loop in a headless browser.">
            <GhostButton onClick={onBuild}><FilePdf size={14} />Build PDF</GhostButton>
          </Empty>
        )}
        {building && !url && <PaperSkeleton width={width * zoom} />}
        {url && !failed && (
          <Document
            key={url}
            file={url}
            onLoadSuccess={({ numPages }) => setPages(numPages)}
            onLoadError={(e) => setFailed(e.message)}
            loading={<PaperSkeleton width={width * zoom} />}
            className="flex w-max min-w-full flex-col items-center gap-4"
          >
            {Array.from({ length: pages }, (_, i) => (
              <Page key={i} pageNumber={i + 1} width={width * zoom} loading={<PaperSkeleton width={width * zoom} />} />
            ))}
          </Document>
        )}
        {failed && <Empty title="Could not open the PDF" body={failed} />}
        {building && url && (
          <div className="pointer-events-none absolute inset-0 rounded-[1.75rem] bg-well/60 backdrop-blur-[2px]" />
        )}
      </div>
    </section>
  );
}

function PaperSkeleton({ width }: { width: number }) {
  return (
    <div className="mx-auto flex flex-col gap-3 rounded bg-white p-8" style={{ width, aspectRatio: "8.5 / 11" }}>
      <div className="shimmer mx-auto h-5 w-1/2 rounded" />
      <div className="shimmer mx-auto h-2.5 w-3/4 rounded" />
      {[0.9, 0.75, 0.95, 0.6, 0.85, 0.7, 0.92, 0.5].map((w, i) => (
        <div key={i} className="shimmer h-2.5 rounded" style={{ width: `${w * 100}%`, marginTop: i % 3 === 0 ? 14 : 0 }} />
      ))}
    </div>
  );
}

function Empty({ title, body, children }: { title: string; body: string; children?: React.ReactNode }) {
  return (
    <div className="flex h-full min-h-[480px] flex-col items-start justify-end gap-3 p-6">
      <div className="mb-auto mt-6 flex aspect-[8.5/11] w-40 flex-col gap-2 rounded bg-white/70 p-4">
        <div className="h-2 w-2/3 rounded bg-zinc-200" />
        <div className="h-1.5 w-full rounded bg-zinc-100" />
        <div className="h-1.5 w-5/6 rounded bg-zinc-100" />
        <div className="mt-2 h-1.5 w-full rounded bg-zinc-100" />
        <div className="h-1.5 w-3/4 rounded bg-zinc-100" />
      </div>
      <h3 className="text-base font-semibold tracking-tight">{title}</h3>
      <p className="max-w-[36ch] text-sm leading-relaxed text-zinc-500">{body}</p>
      {children}
    </div>
  );
}
