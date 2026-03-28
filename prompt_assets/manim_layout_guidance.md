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

7. Use scene-specific box dictionaries.
Declare scene layouts up front as dictionaries such as:
- `SCENE1_BOXES`
- `SCENE2_BOXES`
- `SCENE3_BOXES`

8. Reserve space before animating.
Budget width and height for graphs, braces, formulas, labels, and final answer boxes before adding them.

9. Formula scenes need stricter rules than generic scenes.
- Never keep two dense equations in the same box at the same time.
- Never rely on `.shift()` after `place_in_box(...)` to create a stacked formula layout.
- Instead, create separate formula boxes or a vertical stack helper that arranges formulas inside one parent box.
- Avoid partial indexed transforms between long `MathTex` expressions; use full-group replacements.

Core rule:
fixed scene boxes first
fit objects to their box
clamp to box
then run collision cleanup only as a final safety step
