import { useState } from "react";
import { usePushCode } from "@/api/queries";
import { Actions, Field, Outcome, Run, Text } from "@/components/Form";
import { Panel } from "@/components/Panel";
import { given } from "@/lib/body";
import { errorOf } from "@/lib/error";

/**
 * The two commands that write to the store without queueing anything.
 *
 * They sit apart from Dispatch because neither puts work on the pool: one
 * publishes something a later task will need, the other rewrites the account
 * of tasks already finished.
 *
 * **`push-code` is the one command whose meaning depends on where the server
 * runs.** It seals the working tree, and from a browser "the working tree" is
 * wherever `serve` was launched — not the checkout the operator is looking at.
 * That has already gone wrong once in this project, when `submit` sealed its
 * snapshot from the shell's CWD, so the panel says so and reports back what it
 * actually sealed rather than leaving it implicit.
 */
export function Share() {
  return (
    <div className="space-y-3">
      <PushCode />
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
