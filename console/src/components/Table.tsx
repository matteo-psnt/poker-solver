import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

/** Dense by intent: whitespace that would flatter a marketing page costs a row. */
export function Table({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[12px]">{children}</table>
    </div>
  );
}

export function Th({ children, right }: { children: ReactNode; right?: boolean }) {
  return (
    <th
      className={cn(
        "border-b border-[var(--border)] px-3 py-1.5 font-medium text-[11px] text-[var(--fg-faint)] uppercase tracking-wider",
        right ? "text-right" : "text-left",
      )}
    >
      {children}
    </th>
  );
}

export function Td({
  children,
  right,
  mono,
  className,
}: {
  children: ReactNode;
  right?: boolean;
  mono?: boolean;
  className?: string;
}) {
  return (
    <td
      className={cn(
        "border-b border-[var(--border)]/60 px-3 py-1.5",
        right && "text-right tnum",
        mono && "font-mono",
        className,
      )}
    >
      {children}
    </td>
  );
}
