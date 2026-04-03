import { useLog } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { errorOf } from "@/lib/error";
import { Link, getRouteApi } from "@tanstack/react-router";
import { ArrowLeft } from "lucide-react";

const route = getRouteApi("/legs/$taskId");

/**
 * Why a leg died, which is the question the Legs page raises and could not
 * answer.
 *
 * Reads the copy published to the share, NOT the node's live files: the pool
 * scales to zero within minutes of a task ending, so for exactly the failed
 * legs worth reading the node is already gone. That is what the publish-on-exit
 * trap exists for.
 */
export function LegLog() {
  const { taskId } = route.useParams();
  const log = useLog(taskId);
  const lines = log.data?.lines ?? [];

  return (
    <div className="space-y-3">
      <Link
        to="/legs"
        className="inline-flex items-center gap-1.5 text-[12px] text-[var(--fg-muted)] hover:text-[var(--fg)]"
      >
        <ArrowLeft className="size-3.5" />
        Legs
      </Link>

      <Panel
        title={taskId}
        updatedAt={log.dataUpdatedAt}
        staleAfterMs={Number.POSITIVE_INFINITY}
        error={errorOf(log.error)}
        loading={log.isLoading}
        empty={log.data && lines.length === 0 ? "The published log is empty." : null}
        onRefresh={() => log.refetch()}
        refreshing={log.isFetching}
      >
        {lines.length > 0 && (
          <pre className="max-h-[70vh] overflow-auto px-3 py-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
            {lines.map((line, index) => (
              // Index keys are correct here: this is an immutable, ordered
              // snapshot of a finished log, never reordered or spliced.
              // biome-ignore lint/suspicious/noArrayIndexKey: immutable log snapshot
              <div key={index} className={lineTone(line)}>
                {line}
              </div>
            ))}
          </pre>
        )}
      </Panel>

      <p className="max-w-[70ch] px-1 text-[12px] leading-relaxed text-[var(--fg-muted)]">
        The copy published to the share. The node's own files are gone once the pool scales down,
        which is why this is the durable one.
      </p>
    </div>
  );
}

/** Make the lines that explain a death findable without reading all 400. */
function lineTone(line: string): string {
  const lower = line.toLowerCase();
  if (/\b(error|fatal|traceback|failed|refus)/.test(lower)) return "text-red-400";
  if (/\b(warn|timeout|retry|killed)/.test(lower)) return "text-amber-400";
  if (line.startsWith("[run_leg")) return "text-[var(--fg-faint)]";
  return "";
}
