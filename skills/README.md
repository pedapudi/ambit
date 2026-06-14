# ambit skills

A set of agent skills that
teach an agent (or a new contributor) how to **use** and **develop** ambit. Each
subdirectory is one skill: a `SKILL.md` with `name` / `description` frontmatter and
a focused body. Drop this `skills/` directory where your agent loads project skills
(or point a plugin at it).

| skill | read it when you want to… |
|---|---|
| [`ambit-overview`](ambit-overview/SKILL.md) | understand what ambit *is* and the occupancy mental model — start here |
| [`ambit-cli`](ambit-cli/SKILL.md) | **run** ambit: the `info` / `embed` / `report` commands, every flag, input formats, recipes |
| [`ambit-concepts`](ambit-concepts/SKILL.md) | **read the results**: anisotropy, resolution, and what each diagnostic means |
| [`ambit-architecture`](ambit-architecture/SKILL.md) | understand **how it's built**: the data flow and the core contracts |
| [`ambit-figures`](ambit-figures/SKILL.md) | work on the **report figures**: the figure contract, the registry, adding one |
| [`ambit-development`](ambit-development/SKILL.md) | **contribute**: dev setup, the optional-dependency tiers, code conventions |

## The 30-second version

ambit answers one question: **how much of an embedding space does a dataset
occupy, and where?** It surfaces density hotspots, coverage voids, and the
*resolution* (isotropy) of the space, as a terminal scan or a self-contained HTML
report. One canonical in-memory type (`EmbeddingSet`) flows through a streaming
scan → a shared render context (`Ctx`) → a registry of figures. Core install is
just numpy; heavier capabilities are opt-in extras.
