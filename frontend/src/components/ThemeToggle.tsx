import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Monitor, Moon, Sun } from "@phosphor-icons/react";
import { snappy } from "./ui";

export type Theme = "light" | "dark" | "system";
const KEY = "proofline-theme";
const OPTIONS: { value: Theme; label: string; Icon: typeof Sun }[] = [
  { value: "light", label: "Light theme", Icon: Sun },
  { value: "system", label: "Match system", Icon: Monitor },
  { value: "dark", label: "Dark theme", Icon: Moon },
];

function readSaved(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

function apply(theme: Theme, animate: boolean) {
  const dark = theme === "dark" || (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const root = document.documentElement;
  if (animate) {
    root.classList.add("theme-fade");
    window.setTimeout(() => root.classList.remove("theme-fade"), 400);
  }
  root.classList.toggle("dark", dark);
}

/** Light / system / dark. Saved per browser; "system" follows the OS live. */
export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(readSaved);

  useEffect(() => {
    apply(theme, false);
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => apply("system", true);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [theme]);

  function choose(t: Theme) {
    setTheme(t);
    apply(t, true);
    try {
      if (t === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, t);
    } catch { /* private mode: theme still applies for this visit */ }
  }

  return (
    <div role="radiogroup" aria-label="Color theme"
      className="relative flex items-center rounded-full border border-zinc-200 bg-white p-0.5">
      {OPTIONS.map(({ value, label, Icon }) => {
        const on = theme === value;
        return (
          <button key={value} type="button" role="radio" aria-checked={on} aria-label={label} title={label}
            onClick={() => choose(value)}
            className={`relative flex size-7 items-center justify-center rounded-full transition-colors active:scale-[0.94]
              ${on ? "text-ink" : "text-zinc-400 hover:text-zinc-600"}`}>
            {on && (
              <motion.span layoutId="theme-pill" transition={snappy}
                className="absolute inset-0 rounded-full bg-zinc-100 shadow-[inset_0_0_0_1px_var(--color-zinc-200)]" />
            )}
            <Icon size={14} weight={on ? "fill" : "regular"} className="relative" />
          </button>
        );
      })}
    </div>
  );
}
