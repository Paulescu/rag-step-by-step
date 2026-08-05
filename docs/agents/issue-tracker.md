# Issue tracker: Local Markdown

Issues and PRDs for this repo live as markdown files in `.scratch/`.
These files are tracked in git, deliberately: they are the issue tracker,
so they travel with the code and their history is the audit trail of
triage decisions.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The PRD is `.scratch/<feature-slug>/PRD.md`
- Implementation issues are `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Triage state is recorded as a `Status:` line near the top of each issue file (see `triage-labels.md` for the role strings)
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## Branch naming

A feature directory is an epic and holds several issues, so branch names
echo the *issue* slug, not the feature slug:

    .scratch/conversation-memory/
      PRD.md
      issues/01-store-turns-in-sqlite.md   -> branch store-turns-in-sqlite
      issues/02-summarize-old-turns.md     -> branch summarize-old-turns

This is a convention, not something the skills enforce. It exists so that
a branch name can be grepped back to its issue file and vice versa.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.
