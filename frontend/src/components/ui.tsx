import { memo, useRef, type ReactNode } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";
import type { Status } from "../api";

export const spring = { type: "spring", stiffness: 100, damping: 20 } as const;
export const snappy = { type: "spring", stiffness: 320, damping: 30 } as const;

/** Infinite "breathing" status light. Isolated + memoized so its loop never re-renders a parent. */
export const BreathingDot = memo(function BreathingDot({ tone = "accent" }: { tone?: "accent" | "amber" | "muted" }) {
  const color = tone === "accent" ? "bg-accent" : tone === "amber" ? "bg-amber-500" : "bg-zinc-400";
  return (
    <span className="relative inline-flex size-2">
      <motion.span
        className={`absolute inset-0 rounded-full ${color}`}
        animate={{ scale: [1, 2.2], opacity: [0.45, 0] }}
        transition={{ duration: 1.8, repeat: Infinity, ease: "easeOut" }}
      />
      <span className={`relative inline-flex size-2 rounded-full ${color}`} />
    </span>
  );
});

/** Primary action that leans toward the cursor. Motion values only: no React state per mouse move. */
export function MagneticButton({
  children, onClick, disabled, type = "button", className = "",
}: { children: ReactNode; onClick?: () => void; disabled?: boolean; type?: "button" | "submit"; className?: string }) {
  const ref = useRef<HTMLButtonElement>(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const x = useSpring(mx, { stiffness: 200, damping: 18 });
  const y = useSpring(my, { stiffness: 200, damping: 18 });
  return (
    <motion.button
      ref={ref}
      type={type}
      disabled={disabled}
      onClick={onClick}
      style={{ x, y }}
      onMouseMove={(e) => {
        const r = ref.current?.getBoundingClientRect();
        if (!r || disabled) return;
        mx.set((e.clientX - r.left - r.width / 2) * 0.18);
        my.set((e.clientY - r.top - r.height / 2) * 0.28);
      }}
      onMouseLeave={() => { mx.set(0); my.set(0); }}
      whileTap={{ scale: 0.98 }}
      className={`inline-flex items-center justify-center gap-2 rounded-full bg-ink px-5 py-2.5 text-sm font-medium text-white
        shadow-[inset_0_1px_0_rgba(255,255,255,0.12),0_10px_24px_-12px_rgba(28,28,31,0.55)]
        transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-300 disabled:shadow-none ${className}`}
    >
      {children}
    </motion.button>
  );
}

export function GhostButton({ children, onClick, disabled, title }:
  { children: ReactNode; onClick?: () => void; disabled?: boolean; title?: string }) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium
        text-zinc-700 transition hover:border-zinc-300 hover:text-ink active:scale-[0.98] disabled:opacity-40"
    >
      {children}
    </button>
  );
}

const STATUS: Record<Status, string> = {
  PROVEN: "bg-accent-soft text-accent",
  MISSED: "bg-amber-50 text-amber-700",
  GAP: "bg-rose-50 text-rose-700",
  ROUTED: "bg-zinc-100 text-zinc-500",
};

export function StatusPill({ status }: { status: Status }) {
  return (
    <span className={`inline-flex rounded-full px-2 py-0.5 font-mono text-[10px] font-medium tracking-wide ${STATUS[status]}`}>
      {status}
    </span>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-400">{children}</span>;
}
