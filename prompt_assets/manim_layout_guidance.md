Box-based layout system to prevent overlap:

1. Fixed scene boxes first.
Every major visual region should be declared up front with `layout_box(x, y, width, height)`.
Examples: title band, subtitle band, formula band, graph region, callout region, energy/result region.

2. Fit each object to its assigned box.
Use helper logic equivalent to:
- `fit_to_box(...)`
- `keep_inside_box(...)`
- `place_in_box(...)`

3. Clamp inside the box.
After placement, ensure no edge of the mobject overflows the scene box.

4. Collision-check as a final pass.
Use bounding-box overlap checks equivalent to:
- `mobjects_overlap(...)`
- `resolve_overlap(...)`

5. Avoid primary layout via chained relative positioning.
Do not rely on repeated `.shift()` or `.next_to()` for the major structure of the scene.
Use relative positioning only for small local relationships after the main boxes are established.

6. Prefer fade-based swaps in tight regions.
For crowded formula regions or subtitles, use fade swaps instead of large in-place transforms.
Pattern:
- `fade_swap(old_mob, new_mob)`
- Better for reusable formula bands:
- `replace_in_box(old_group, new_group, formula_box)`
- or an equivalent explicit box-owner swap helper

7. Use scene-specific box dictionaries.
Declare scene layouts up front as dictionaries such as:
- `SCENE1_BOXES`
- `SCENE2_BOXES`
- `SCENE3_BOXES`

8. Reserve space before animating.
Budget width and height for graphs, braces, formulas, labels, and final answer boxes before adding them.

8b. Graph scenes need explicit structural marks.
- If a graph uses axes, label the x-axis and y-axis clearly.
- If the problem names vertical boundaries like `x = 1` and `x = 4`, those lines should be drawn at the actual coordinates, not just mentioned as floating text.
- If a bounded region is shaded, make sure the curve, x-axis, and every stated boundary line are all visible together.
- If the problem says "area under the curve," the shaded region must close against the x-axis or another explicitly stated lower boundary, not against a diagonal chord.
- Use `axes.get_area(...)` only with a function graph from `axes.plot(...)`. If the visible curve is parametric, convert the relevant part to a function graph or build the region boundary explicitly.

8c. Readability is part of layout.
- Do not solve crowding by shrinking important text to tiny sizes.
- Main formulas, evaluation lines, and explanatory prose should stay comfortably readable.
- If content does not fit at a readable size, give it a larger box or split it across more scenes.
- Choose box heights from line count. A derivation stack with 3 or 4 lines needs a genuinely tall derivation band; do not trust `fit_to_box(...)` to save an undersized layout.
- Multi-line answer cards and summary panels need extra vertical room for text plus padding and any surrounding rectangle.
- If a box is short enough that auto-fit would noticeably miniaturize the content, the layout is wrong and should be restructured.

8d. Visual density matters too.
- A clean layout should not become an empty layout. Avoid scenes where only one small formula or number sits alone on a mostly blank frame.
- Every scene should either let one major visual region carry the frame with real presence, such as a graph, diagram, or large derivation stack, or keep at least two purposeful regions active, such as title plus derivation, graph plus callout, or answer plus recap note.
- Final answer scenes should use a result panel, summary strip, or paired callout with context like units, exact form, interpretation, or a compact recap. Do not end on a tiny centered `MathTex` line by itself unless it is a very brief transition.
- When a derivation resolves to a single value, promote that value into a larger answer card or keep a compact supporting stack visible until the answer panel is fully established.

9. Formula scenes need stricter rules than generic scenes.
- Never keep two dense equations in the same box at the same time.
- Never rely on `.shift()` after `place_in_box(...)` to create a stacked formula layout.
- Instead, create separate formula boxes or a vertical stack helper that arranges formulas inside one parent box.
- Every major equation state that appears in `FadeIn`, `Transform`, or `ReplacementTransform` must be placed in its destination box before the animation begins.
- If a formula box is reused across successive scenes, the old box occupant must explicitly leave or be replaced before the new group is revealed.
- Do not keep an intermediate heading like `Antiderivative:` or an earlier formula visible underneath later evaluation lines in the same box.
- For sequential evaluation, replace the old formula group with a new evaluation container first, then reveal lines within that container.
- Build the new evaluation container with `VGroup(...).arrange(DOWN, ...)` before it enters the box, so the box owner is a single arranged group instead of several floating lines.
- If braces or coefficient labels are needed, reserve a separate annotation band below or beside the equation instead of attaching them inside the same dense formula box.
- Do not create braces and labels with `next_to(...)` and then move the whole brace+label group into another box; build the braces from the already-positioned equation and place the labels in their own box or row.
- Avoid partial indexed transforms between long `MathTex` expressions; use full-group replacements.

Core rule:
fixed scene boxes first
fit objects to their box
clamp to box
then run collision cleanup only as a final safety step
