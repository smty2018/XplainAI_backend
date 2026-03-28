Strict no-overlap rules for generated Manim code:

1. One major content group per box at a time.
- A formula box may show one full equation group at once.
- Before a new full equation enters the same box, the old one must fully leave or be replaced in-place.

2. Never use `.shift()` as the primary way to separate major objects after `place_in_box(...)`.
- If two objects need distinct vertical positions, create two distinct boxes.
- Use box dictionaries such as `FORMULA_TOP_BOX`, `FORMULA_MID_BOX`, `FORMULA_BOTTOM_BOX`.

3. Do not transform indexed submobjects between dense equations when both equations are long.
- Avoid patterns like `Transform(expr1[2], expr2[4])` for long formulas.
- Prefer whole-group `fade_swap(...)`, `ReplacementTransform(...)`, or staged multi-line replacements.

4. Long equations must be reflowed before display.
- If a formula exceeds about 80 percent of the available box width, split it into two or more lines.
- Use a helper such as `stack_in_box(...)` or `multiline_formula(...)`.

5. Add dedicated helpers for formula layout.
- `stack_in_box(mobs, box, gap=...)`
- `replace_in_box(old_mob, new_mob, box, ...)`
- `mobjects_overlap(mob_a, mob_b, gap=...)`
- `resolve_overlap(mob, blockers, box, gap=..., step=...)`
- `layout_pass(mobs, box, blockers=...)`

6. Re-run layout after every major state change.
- After placing, replacing, or transforming major groups, re-fit to the box and resolve overlaps before the next animation.
- Collision checking is mandatory, not optional. The final script should contain a `mobjects_overlap(...)` check and a `resolve_overlap(...)` repair pass similar to the tested layout framework.

7. Favor sequential visibility over simultaneous density.
- In algebra scenes, do not keep the old factorization, the distributed expansion, and the combined result all visible in the same region unless each has its own dedicated box.

8. Final self-check before returning code.
- No two visible major groups overlap.
- No formula spills outside its box.
- No dense formula is manually nudged by repeated `.shift(...)` after initial placement.
- No crowded same-box expression uses partial-index transforms when a safer full-group replacement is possible.
