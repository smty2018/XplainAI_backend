# Box layout and collision prevention

Use fixed layout boxes before placing any major object.

Recommended helpers:

- `layout_box(x, y, width, height)`
- `fit_to_box(...)`
- `keep_inside_box(...)`
- `place_in_box(...)`
- `mobjects_overlap(...)`
- `resolve_overlap(...)`

Core rule:

1. define scene boxes first
2. fit each object to its box
3. clamp objects inside the box
4. only then run collision cleanup as a final safety step

Avoid using repeated `.shift(...)` calls as the main layout strategy for titles, formulas, graphs, or answer boxes.

Prefer separate boxes for:

- title
- subtitle
- formula area
- graph area
- side callouts
- final answer

When crowded formulas need to change, use a fade-based or replacement-style swap inside the same box instead of stacking overlapping equations.
