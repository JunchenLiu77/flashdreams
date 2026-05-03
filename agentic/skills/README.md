# Agentic skills

Project-authored [Agent Skills](https://docs.cursor.com/agent/skills) for this repository. These are **opt-in**: neither Cursor nor Claude Code loads skills from `agentic/skills/` automatically, so they won't collide with anything a developer already has under `~/.cursor/skills/` or `~/.claude/skills/`.

Opt in by symlinking this directory into whichever tool you use. From the repo root:

```bash
# Cursor
mkdir -p .cursor && ln -s ../agentic/skills .cursor/skills

# Claude Code
mkdir -p .claude && ln -s ../agentic/skills .claude/skills
```

`.cursor/` and `.claude/` are already gitignored (add entries if they aren't), so the symlinks stay local. If you want to be selective, symlink individual skill directories instead of the whole folder:

```bash
ln -s ../../agentic/skills/<skill-name> .cursor/skills/<skill-name>
```

## Layout

Each skill is a directory containing a `SKILL.md` with YAML frontmatter:

```
agentic/skills/
├── README.md
└── <skill-name>/
    ├── SKILL.md          # required: name, description, body
    ├── reference.md      # optional: deeper docs, linked from SKILL.md
    └── scripts/          # optional: executable helpers
```

The `SKILL.md` format is the same for Cursor and Claude Code: YAML frontmatter with `name` (lowercase-hyphenated, ≤64 chars) and `description` (specific, third person, includes both *what* and *when*), followed by markdown instructions. Keep the body under ~500 lines and push long-form reference into sibling files.

## Authoring

When adding a skill, prefer codifying conventions already visible in the codebase over re-deriving them. The goal is to reduce drift between human-written code and agent-written code, not to invent new style.
