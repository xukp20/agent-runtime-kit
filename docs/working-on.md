# Active work index

Keep `dev_docs/working_on.md` as a short coordination index, not a task log.

## Format

Use one table with one row per task and no duplicate detailed sections:

| Task ID | Status / owner | Goal and next action | Write scope / conflict | Resume link | Updated |
| --- | --- | --- | --- | --- | --- |

Use a short sentence per cell. Aim for at most 200 Chinese characters (or about
100 English words) per row, excluding paths and links. Put the full brief,
participants, constraints, verification, and evidence in the linked task package.

Do not append event logs, historical progress, session or lease identifiers,
process IDs, hashes, or test transcripts to the index.

## Lifecycle

- Register work before writing. Read the current index, linked brief, and Git
  status; preserve other contributors' changes and coordinate overlapping scope.
- Include only current execution, actionable blockers, or a concrete near-term
  wait condition with an owner. Keep unscheduled proposals and long-term waits
  in a backlog or deferred-work document.
- Update only your row. A designated coordinator may maintain shared entries.
  Do not rewrite other tasks unless the user authorizes a coordinated cleanup.
- On completion, save the result, verification, and remaining work in the task
  record, then remove the active row in the same turn. Do not leave completed
  work indefinitely marked as closing.
- Review your entry when resuming and delivering work. Age alone does not prove
  completion. Preserve uncertain tasks in an explicitly unverified queue with
  their owner and resume link; do not cancel tasks or goals by moving a row.
- Before an authorized historical cleanup, archive the complete previous index
  and record the classification rationale. Keep the underlying evidence intact.
- An empty active table is valid. Local process documents remain untracked
  unless the user explicitly asks to version them.

The linked task package is the source of detailed recovery information.
