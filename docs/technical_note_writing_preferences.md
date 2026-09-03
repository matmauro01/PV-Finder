# Technical Note Writing Preferences

This file records the agreed style for revising the ATLAS PV-Finder technical
note. Read it before reviewing or editing another chapter.

## Goal

Write in Mattia's own voice: a graduate student explaining work he performed.
The prose should be technically precise, plain, and direct. It should not sound
like advertising or like a polished AI summary.

## Style

- Use short sentences, usually with one main idea each.
- Prefer simple verbs: `uses`, `shows`, `gives`, `finds`, `increases`, `removes`.
- State the measurement first. Give only the interpretation needed to understand it.
- Keep numbers, definitions, caveats, citations, and LaTeX notation accurate.
- Use `we` when it is natural, but do not begin every paragraph with it.
- Keep paragraph transitions simple. Some variation in sentence length is fine.
- Use cautious wording when the evidence is not absolute: `suggests`, `is consistent
  with`, or `is difficult to separate`.
- Do not introduce intentional mistakes, informality, or vague language to appear human.

Avoid repeated rhetorical constructions and polished slogans, including:

- `not X, but Y` or `rather than X` when two direct sentences would be clearer;
- `the central result`, `the natural next step`, `sets the next steps`;
- `the difference is instructive rather than disappointing`;
- headings such as `Quote the margin, not the absolute`;
- instructions telling the reader how a result "must" or "should" be read;
- claims such as `definitive`, `physical limit`, or `unresolvable` unless they are
  strictly demonstrated;
- em dashes and long sentences joined by several semicolons or parenthetical clauses.

Overview sections should omit model-version labels, file tags, and implementation
details unless they are needed to understand the result.

## Examples

Too polished:

> The difference is instructive rather than disappointing: the floor has already
> taken the easy fakes and the gate is left with a harder residual.

Preferred:

> After the height cut, the classifier removes another 9.5% of the fake vertices.
> The improvement is smaller because the two selections use similar information.

Too rhetorical:

> The remaining headroom is the finder's, by a factor of four.

Preferred:

> The truth-guided assignment reaches 73.7% with a 0.5 mm window. The GNN
> reaches 68.4%, which is 5.2 percentage points lower. The truth-guided result
> is a diagnostic reference, not a strict upper bound.

Too strong:

> These close pairs are unresolvable and form a physical ceiling.

Preferred:

> About 13% of adjacent truth-vertex pairs are separated by less than the measured
> PV-Finder vertex--vertex resolution of 0.229 mm. These pairs are particularly
> difficult to separate.

## Review workflow

1. Read the full chapter and check it against the current results.
2. Separate necessary factual updates from optional language changes.
3. Show the proposed wording to Mattia before editing.
4. After approval, edit the source and make one small local commit for that chapter.
5. Do not compile, regenerate the PDF, or push after every chapter.
6. At the end, review the complete diff, compile once, inspect the PDF, and push the
   accumulated commits to GitLab.

The canonical note checkout is
`/data/home/matmauro/codice/ANA-IDTR-2026-01-INT1`. The revisions through the
GNN chapter were compiled and pushed to GitLab on 3 September 2026.

For a long or number-heavy chapter, a subagent may independently check facts,
tables, and consistency. The primary agent remains responsible for the wording so
the voice stays consistent.
