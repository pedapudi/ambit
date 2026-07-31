# ambit skills

A set of focused, **agent-agnostic** skills that teach an agent — or a new human
contributor — how to **use** and **develop** ambit. Each subdirectory is one skill:
a `SKILL.md` with a short `name` / `description` and a self-contained body. They are
plain Markdown — load them into whatever coding agent you use, or just read them.
See [`AGENTS.md`](../AGENTS.md) for the project's top-level agent guide.

| skill | read it when you want to… |
|---|---|
| [`ambit-overview`](ambit-overview/SKILL.md) | understand what ambit *is* and the occupancy mental model — start here |
| [`ambit-cli`](ambit-cli/SKILL.md) | **run** ambit: the `info` / `embed` / `report` commands, every flag, input formats, recipes |
| [`ambit-concepts`](ambit-concepts/SKILL.md) | **read the results**: anisotropy, resolution, and what each diagnostic means |
| [`ambit-architecture`](ambit-architecture/SKILL.md) | understand **how it's built**: the data flow and the core contracts |
| [`ambit-figures`](ambit-figures/SKILL.md) | work on the **report figures**: the figure contract, the registry, adding one |
| [`ambit-tuning`](ambit-tuning/SKILL.md) | **fix** what the report finds: set up a measurement-driven fine-tune — diagnose, mine, regularize, verify |
| [`ambit-development`](ambit-development/SKILL.md) | **contribute**: dev setup, the optional-dependency tiers, code conventions |

## The 30-second version

ambit answers one question: **where is an embedded dataset too crowded to keep
its items apart — and which items are in trouble?** It measures occupancy
continuously (no bins), judges every number against a well-spread reference,
and names the affected entities by id — as a terminal scan, a self-contained
HTML report that reads as a story, a two-embedding comparison, and
training-time losses that aim gradient at the measured crowding. One canonical
in-memory type (`EmbeddingSet`) flows through a streaming scan → a shared
render context (`Ctx`) → a registry of figures. Core install is just numpy;
heavier capabilities are opt-in extras.
