import { useCompactLegs, usePushCode, usePushData } from "@/api/queries";
import { Actions, Field, Guard, Outcome, Run, Text } from "@/components/Form";
import { Panel } from "@/components/Panel";
import { given } from "@/lib/body";
import { errorOf } from "@/lib/error";
import { count } from "@/lib/format";
import { useState } from "react";

/**
 * The three commands that write to the share without queueing anything.
 *
 * They sit apart from Dispatch because none of them puts work on the pool: two
 * publish something a later task will need, and the third rewrites the account
 * of tasks already finished.
 *
 * **These are the only commands whose meaning depends on where the server
 * runs.** `push-code` seals the working tree and `push-data` copies local
 * abstractions, and from a browser "the working tree" is wherever `serve` was
 * launched — not the checkout the operator is looking at. That has already gone
 * wrong once in this project, when `submit` sealed its snapshot from the
 * shell's CWD, so both panels say so and both report back what they actually
 * sealed rather than leaving it implicit.
 */
export function Share() {
  return (
    <div className="space-y-3">
      <PushCode />
      <PushData />
      <CompactLegs />
    </div>
  );
}

/** `push-code`. An immutable snapshot of a tree, echoing its id. */
function PushCode() {
  const push = usePushCode();
  const [root, setRoot] = useState("");

  return (
    <Panel title="Push code — snapshot the working tree">
      <div className="divide-y divide-[var(--border)]/50">
        <Field
          label="root"
          hint="Blank snapshots the checkout this SERVER is running from, which is not necessarily the one you are editing. Give a path to be sure."
        >
          <Text value={root} onChange={setRoot} placeholder="(the server's checkout)" />
        </Field>
      </div>
      <Actions note="A snapshot is immutable; pushing again makes a new one rather than replacing it.">
        <Run
          label="Push code"
          pending={push.isPending}
          onClick={() => push.mutate(given({ root }))}
        />
      </Actions>
      <Outcome error={errorOf(push.error)}>
        {push.data && <span>sealed {push.data.code_snapshot}</span>}
      </Outcome>
    </Panel>
  );
}

/**
 * `push-data`. Abstractions are COPIED to the share, never recomputed on a
 * node — which is the same rule that says nothing on the laptop reads a card
 * abstraction to decide something about the wrong machine.
 */
function PushData() {
  const push = usePushData();
  const [source, setSource] = useState("");
  const [name, setName] = useState("");

  const uploaded = Object.entries(push.data?.uploaded ?? {});

  return (
    <Panel title="Push data — publish card abstractions">
      <div className="divide-y divide-[var(--border)]/50">
        <Field label="source" hint="Blank uses the server's default abstractions directory.">
          <Text value={source} onChange={setSource} placeholder="(default)" />
        </Field>
        <Field label="name" hint="Publish one abstraction directory. Blank publishes all of them.">
          <Text value={name} onChange={setName} placeholder="(all)" />
        </Field>
      </div>
      <Actions>
        <Run
          label="Push data"
          pending={push.isPending}
          onClick={() => push.mutate(given({ source, name }))}
        />
      </Actions>
      <Outcome error={errorOf(push.error)}>
        {push.data &&
          (uploaded.length === 0 ? (
            // Not an error and not nothing: the share already had every file,
            // which is the ordinary result of pushing twice.
            <span className="text-[var(--fg-muted)]">
              Nothing to upload — the share is already current.
            </span>
          ) : (
            <div className="space-y-0.5">
              {uploaded.map(([abstraction, files]) => (
                <div key={abstraction}>
                  {abstraction} <span className="text-[var(--fg-faint)]">{count(files)} files</span>
                </div>
              ))}
            </div>
          ))}
      </Outcome>
    </Panel>
  );
}

/**
 * `compact-legs`, in the shape the command itself argues for.
 *
 * The dry run is the default and always runs first: it reports what WOULD move
 * without touching anything, and only then is applying offered. `--delete` is
 * the irreversible half and the only one that buys the speedup, so it is a
 * guard flag and it requires a backup path — which the command requires too,
 * because nothing else holds a copy of the record.
 */
function CompactLegs() {
  const compact = useCompactLegs();
  const [apply, setApply] = useState(false);
  const [remove, setRemove] = useState(false);
  const [backup, setBackup] = useState("");
  const [label, setLabel] = useState("");

  // The command refuses `--delete` without `--backup`. Blocking it here means
  // finding out before a share read rather than after one.
  const ready = !remove || Boolean(backup.trim());

  return (
    <Panel title="Compact legs — bundle sealed task records">
      <div className="divide-y divide-[var(--border)]/50">
        <Field label="label" hint="Names the bundle: <label>.bundle.json.">
          <Text value={label} onChange={setLabel} placeholder="sealed" />
        </Field>
        <Field
          label="backup"
          hint="Where every leg file is copied before anything is deleted. Required with --delete."
        >
          <Text value={backup} onChange={setBackup} placeholder="/path/to/legs-backup" />
        </Field>
      </div>

      <div className="border-t border-[var(--border)] divide-y divide-[var(--border)]/50">
        <Guard
          label="--apply"
          because="write the bundle. Without this the run only reports what would move."
          checked={apply}
          onChange={(next) => {
            setApply(next);
            // Deleting without applying is not a state the command has: the
            // bundle has to exist and verify before the files it replaced can
            // go. Un-arming both together keeps the form from offering it.
            if (!next) setRemove(false);
          }}
        />
        <Guard
          label="--delete"
          because="remove the loose files the bundle replaced, after it verifies. The irreversible half, and the only one that buys the speedup."
          checked={remove}
          disabled={!apply}
          onChange={setRemove}
        />
      </div>

      <Actions
        note={
          !apply
            ? "Dry run: reports what would move and changes nothing."
            : !ready
              ? "A backup path is required before deleting."
              : null
        }
      >
        <Run
          label={apply ? (remove ? "Bundle and delete" : "Bundle") : "Preview"}
          danger={remove}
          pending={compact.isPending}
          disabled={!ready}
          onClick={() => compact.mutate(given({ apply, delete: remove, backup, label }))}
        />
      </Actions>

      <Outcome error={errorOf(compact.error)}>
        {compact.data && (
          <div className="space-y-0.5">
            <div>
              {compact.data.files_before} files → {compact.data.files_after}
              <span className="ml-2 text-[var(--fg-faint)]">
                {count(compact.data.movable)} movable
                {compact.data.carried > 0 && ` · ${count(compact.data.carried)} already bundled`}
              </span>
            </div>
            {compact.data.bundle && (
              <div className="text-[var(--fg-muted)]">bundle {compact.data.bundle}</div>
            )}
            {/* Verified is not decoration: the delete is gated on it, and a
                bundle that wrote but did not verify is exactly the state where
                nothing further should happen. */}
            <div className="text-[var(--fg-faint)]">
              {compact.data.applied ? "applied" : "dry run"}
              {compact.data.applied && (compact.data.verified ? " · verified" : " · NOT VERIFIED")}
              {compact.data.deleted > 0 && ` · deleted ${count(compact.data.deleted)}`}
              {compact.data.backup && ` · backup ${compact.data.backup}`}
            </div>
          </div>
        )}
      </Outcome>
    </Panel>
  );
}
