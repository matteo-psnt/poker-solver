import { AlertTriangle } from "lucide-react";
import { cloneElement, type ReactElement, type ReactNode, useId } from "react";
import { cn } from "@/lib/utils";

/**
 * The controls the dispatching pages are built from.
 *
 * Deliberately shaped like the rest of the console rather than like a web form:
 * 12px, mono where the value is an identifier, a bordered row per field, no
 * floating labels and no tinted cards. A dispatch page that read as a different
 * application would be exactly the drift the row-not-card decision elsewhere
 * avoids.
 *
 * Every control is CONTROLLED and holds a STRING, including the numeric ones. An
 * `<input type=number>` whose value is a number cannot represent "the field is
 * empty", and empty is the state that means *omit this argument* -- which is the
 * contract the server's request bodies rest on.
 */

/**
 * A labelled row. The hint is the flag's own help, shortened.
 *
 * The association is EXPLICIT — `htmlFor` on the label, the same generated id
 * cloned onto the control. Wrapping the control in the label would associate
 * them implicitly and read more simply, but a label whose control arrives as
 * `{children}` cannot be checked by anything: neither the linter nor a reader
 * can see whether there is a control in there at all. Explicit is also what
 * makes the hint clickable-safe — it sits outside the control, and an implicit
 * label would make clicking the help text focus the field.
 */
export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactElement<{ id?: string }>;
}) {
  const id = useId();
  return (
    <div className="grid grid-cols-[7.5rem_1fr] items-baseline gap-3 px-3 py-1.5">
      <label
        htmlFor={id}
        className="pt-1 text-[11px] text-[var(--fg-faint)] uppercase tracking-wider"
      >
        {label}
      </label>
      <div className="min-w-0">
        {cloneElement(children, { id })}
        {hint && <p className="mt-1 text-[11px] text-[var(--fg-faint)]">{hint}</p>}
      </div>
    </div>
  );
}

const CONTROL =
  "w-full rounded border border-[var(--border)] bg-transparent px-2 py-1 text-[12px] " +
  "text-[var(--fg)] outline-none placeholder:text-[var(--fg-faint)] " +
  "focus:border-[var(--fg-faint)] disabled:opacity-40";

export function Text({
  id,
  value,
  onChange,
  placeholder,
  mono = true,
  disabled,
}: {
  id?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  mono?: boolean;
  disabled?: boolean;
}) {
  return (
    <input
      id={id}
      type="text"
      value={value}
      disabled={disabled}
      placeholder={placeholder}
      onChange={(event) => onChange(event.target.value)}
      className={cn(CONTROL, mono && "font-mono")}
    />
  );
}

/**
 * A numeric field that is still a string.
 *
 * `inputMode` rather than `type="number"`: the spinner is useless at these
 * magnitudes (an iteration target is 25,000,000) and the browser's own
 * validation would silently blank a value it disliked, which is the one thing a
 * field feeding a dispatch must never do.
 */
export function Num({
  id,
  value,
  onChange,
  placeholder,
  disabled,
}: {
  id?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
}) {
  return (
    <input
      id={id}
      type="text"
      inputMode="numeric"
      value={value}
      disabled={disabled}
      placeholder={placeholder}
      onChange={(event) => onChange(event.target.value)}
      className={cn(CONTROL, "tnum font-mono")}
    />
  );
}

/**
 * A picker. The empty option is a real choice — it means "leave this flag
 * unset" — so it is offered rather than defaulted away.
 */
export function Select({
  id,
  value,
  onChange,
  options,
  placeholder = "—",
  disabled,
}: {
  id?: string;
  value: string;
  onChange: (value: string) => void;
  options: readonly string[];
  placeholder?: string;
  disabled?: boolean;
}) {
  return (
    <select
      id={id}
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
      className={cn(CONTROL, "font-mono")}
    >
      <option value="">{placeholder}</option>
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}

/**
 * A flag whose default is off and must stay that way.
 *
 * Every one of these carries a documented reason to be off: `--force` on a
 * precompute "silently invalidates the provenance of every run trained against
 * it", `--force` on a compare produces a p-value that is not trustworthy,
 * `--delete` on a compact is the irreversible half. So the reason is shown
 * beside the box rather than in a tooltip, and the box is never pre-checked —
 * the server's body model omits an unchecked flag entirely, which lets the
 * command's own default answer.
 */
export function Guard({
  label,
  because,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  because: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2 px-3 py-1.5">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5 size-3.5 accent-[#E0655C]"
      />
      <span className="min-w-0">
        <span className={cn("font-mono text-[12px]", checked && "text-[#E0655C]")}>{label}</span>
        <span className="ml-2 text-[11px] text-[var(--fg-faint)]">{because}</span>
      </span>
    </label>
  );
}

/**
 * The button that does it.
 *
 * `pending` disables rather than hides: a dispatch seals a code snapshot and
 * queues a task, which is 10-20 seconds, and a control that vanished would
 * leave the page looking like nothing happened. Disabling is also the only
 * defence against a double-click queueing two runs — the server cannot tell one
 * from a deliberate second submission and should not try.
 */
export function Run({
  label,
  onClick,
  pending,
  disabled,
  danger,
}: {
  label: string;
  onClick: () => void;
  pending?: boolean;
  disabled?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || pending}
      className={cn(
        "rounded border px-3 py-1 font-mono text-[12px] transition-colors disabled:opacity-40",
        danger
          ? "border-[#E0655C]/60 text-[#E0655C] hover:bg-[#E0655C]/10"
          : "border-[var(--fg-faint)] text-[var(--fg)] hover:bg-white/[0.06]",
      )}
    >
      {pending ? "working…" : label}
    </button>
  );
}

/** The row a form's controls sit in, so every page's footer lines up. */
export function Actions({ children, note }: { children: ReactNode; note?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-3 border-t border-[var(--border)] px-3 py-2">
      {children}
      {note && <span className="text-[11px] text-[var(--fg-faint)]">{note}</span>}
    </div>
  );
}

/**
 * What came back, or why nothing did.
 *
 * Rendered inside the panel rather than as a toast. A dispatch's answer is a
 * job id and a task list — things to be read and correlated with the task
 * table, not glanced at while they fade.
 */
export function Outcome({ error, children }: { error?: string | null; children?: ReactNode }) {
  if (!error && !children) return null;
  return (
    <div
      className={cn(
        "border-t px-3 py-2 font-mono text-[12px]",
        error ? "border-red-500/30 bg-red-500/5 text-red-400" : "border-[var(--border)]",
      )}
    >
      {error ? (
        <span className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          {error}
        </span>
      ) : (
        children
      )}
    </div>
  );
}

/** What a dispatch queued: the snapshot it sealed, the job, and the tasks. */
export function Queued({
  snapshot,
  job,
  tasks,
}: {
  snapshot: string;
  job: string;
  tasks: string[];
}) {
  return (
    <div className="space-y-0.5">
      <div>
        queued {tasks.length} task{tasks.length === 1 ? "" : "s"} into{" "}
        <span className="text-[var(--fg)]">{job}</span>
      </div>
      <div className="text-[var(--fg-faint)]">snapshot {snapshot}</div>
      {tasks.map((task) => (
        <div key={task} className="text-[var(--fg-muted)]">
          {task}
        </div>
      ))}
    </div>
  );
}
