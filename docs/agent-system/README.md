# Research agent setup

This project is configured so expensive frontier reasoning is concentrated on scientific decisions while routine implementation is delegated.

## OpenCode roles

- `lab` — default Terra supervisor. Use for normal operation, task routing, execution, and experiment bookkeeping.
- `research` — Sol research collaborator. Use for free-form hypothesis discussion, mathematical caveats, experiment design, and interpretation.
- `research-engineer` — Terra implementation worker for substantial numerical/multi-file tasks.
- `routine-worker` — Luna worker for well-specified mechanical coding, tests, plots, data plumbing, and ordinary debugging.
- `scientific-critic` — read-only Terra methodological adversary.
- `sol-reviewer` — read-only Sol escalation for genuinely consequential unresolved issues.

OpenCode V2 discovers these agents from `.opencode/agents/`. Use `/agents` or the TUI agent selector to switch primary agents. The `research` agent can delegate a frozen experiment to `lab`; `lab` can delegate to the workers and escalate to `sol-reviewer`.

## Recommended conversation pattern

For research ideation, switch to `research` and talk normally. The point is to preserve the existing high-value Sol collaboration: explore hypotheses, ask what is wrong with an idea, look for hidden assumptions, and design decisive experiments.

Once the idea is sufficiently precise, ask the research agent to freeze an implementation-ready research contract and delegate it to `lab`.

For routine operation, use `lab`. It should delegate mechanical work rather than perform it itself. If a worker encounters a scientific ambiguity, the task returns upward rather than being guessed away.

## Verification philosophy

Workers are not trusted because of model reputation. They are trusted to the extent that their task is objectively checkable.

Use tests, synthetic recovery, invariance/equivalence checks, null behavior, deterministic seeds, lint, exact artifacts, and independent review. Escalate when correctness cannot be made observable.

## Current provider setup

The repository does not contain credentials. OpenAI/ChatGPT authentication remains in your local OpenCode installation.

The initial model hierarchy uses only the OpenAI models already available through your connected ChatGPT Plus/Pro account:

- Sol for research and expensive escalation;
- Terra for supervision, scientific criticism, and substantial engineering;
- Luna for routine implementation.

A cheap external provider such as DeepSeek can be added later with `/connect`. After that, selected worker agent files can simply change their `model:` line; the research memory and orchestration design do not need to change.

## Safety defaults

The project config:

- permits repository reading, searching, web search/fetch, and skills;
- asks before arbitrary shell commands;
- permits common read-only Git commands;
- denies automatic `git push`, destructive reset/clean, and common direct deletion commands.

Worker agents explicitly allow the normal pytest/Ruff verification commands. Widen permissions only after observing real tasks and deciding which commands are safe for unattended execution.

## Durable memory

Scientific state is not stored only in chat history:

- `RESEARCH_PROGRAM.md` — enduring objective and standards;
- `CURRENT_RESEARCH_STATE.md` — current best understanding;
- `docs/HYPOTHESIS_LEDGER.md` — hypotheses and their status;
- `docs/experiments/EXPERIMENT_INDEX.md` — compact anti-duplication record;
- `docs/checkpoints/` — detailed historical evidence;
- Git — exact implementation history;
- `.opencode/skills/` — reusable procedural/epistemic knowledge.

The experiment index currently seeds checkpoint 0026; the older checkpoint history should be backfilled automatically as a bookkeeping task. Until then, agents are required to search the checkpoint directory before calling an experiment new.

## Next layers

After this OpenCode configuration works in real use:

1. connect a cheap external provider and redirect routine workers if desired;
2. install Hermes as the persistent conversational supervisor above OpenCode;
3. import/extract the high-value state from the current Codex/Sol conversation into these durable files and skills;
4. backfill the experiment index from checkpoints 0001–0026;
5. add an experiment tracker such as MLflow if run volume warrants it;
6. add Modal/other rented GPU compute with hard spend/termination rules;
7. widen unattended permissions only after the workflow has been observed safely;
8. later evaluate which model/skill combinations actually catch real research mistakes and optimize routing empirically.

The architecture is deliberately incremental: the durable scientific state and role boundaries should survive future changes in model provider, harness, GPU platform, or orchestration layer.
