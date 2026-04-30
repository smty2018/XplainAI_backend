Strict no-overlap rules for generated Manim code:

1. One major content group per box at a time.
- A formula box may show one full equation group at once.
- Before a new full equation enters the same box, the old one must fully leave or be replaced in-place.
- Reusing the same box across scenes is allowed only when the transition explicitly removes or replaces the previous occupant.

2. Never use `.shift()` as the primary way to separate major objects after `place_in_box(...)`.
- If two objects need distinct vertical positions, create two distinct boxes.
- Use box dictionaries such as `FORMULA_TOP_BOX`, `FORMULA_MID_BOX`, `FORMULA_BOTTOM_BOX`.

3. Do not transform indexed submobjects between dense equations when both equations are long.
- Avoid patterns like `Transform(expr1[2], expr2[4])` for long formulas.
- Prefer whole-group `fade_swap(...)`, `ReplacementTransform(...)`, or staged multi-line replacements.

4. Long equations must be reflowed before display.
- If a formula exceeds about 80 percent of the available box width, split it into two or more lines.
- Use a helper such as `stack_in_box(...)` or `multiline_formula(...)`.
- If 3 or more lines are stacked into one derivation group, the destination box must be tall enough that the group can stay readable after placement.
- Do not hide a bad box budget behind auto-scaling. If the group only fits by shrinking to a tiny size, the scene should be split or the box should be enlarged.

4b. Every transformed formula state must be placed before animation.
- If `step1`, `step2`, `eq_standard`, or similar mobjects are used in `FadeIn`, `Transform`, or `ReplacementTransform`, each one must be explicitly placed in its destination box before it is animated.
- Do not assume a newly created `MathTex` will inherit the correct layout just because it is part of a transform chain.

4c. Brace annotations need their own band.
- Do not keep braces, coefficient labels, and a dense equation in the same visual band.
- Build braces from the already-positioned equation, then place labels in a dedicated annotation box or separate horizontal row.
- Avoid `next_to(...)` on brace labels followed by moving the entire brace+label group into another box.

5. Add dedicated helpers for formula layout.
- `stack_in_box(mobs, box, gap=...)`
- `replace_in_box(old_mob, new_mob, box, ...)`
- `clear_box_owner(old_mob, ...)` or `swap_box_owner(old_mob, new_mob, box, ...)`
- `mobjects_overlap(mob_a, mob_b, gap=...)`
- `resolve_overlap(mob, blockers, box, gap=..., step=...)`
- `layout_pass(mobs, box, blockers=...)`

6. Re-run layout after every major state change.
- After placing, replacing, or transforming major groups, re-fit to the box and resolve overlaps before the next animation.
- Collision checking is mandatory, not optional. The final script should contain a `mobjects_overlap(...)` check and a `resolve_overlap(...)` repair pass similar to the tested layout framework.

7. Favor sequential visibility over simultaneous density.
- In algebra scenes, do not keep the old factorization, the distributed expansion, and the combined result all visible in the same region unless each has its own dedicated box.
- In calculus evaluation scenes, do not keep `Antiderivative:` text, the primitive function, and the substitution stack visible in one formula band unless each has a separate box.
- When a box changes owners, treat that as a scene transition event: old owner out, new arranged owner in, then reveal details inside the new owner.
- Sequential visibility does not mean an underfilled frame. Do not clear away the supporting context so aggressively that one small leftover formula is stranded on a blank screen.
- If only one result line remains, promote it into a larger result panel or keep a nearby title, recap note, exact form, or supporting derivation summary visible with it.

8. Final self-check before returning code.
- No two visible major groups overlap.
- No formula spills outside its box.
- No dense formula is manually nudged by repeated `.shift(...)` after initial placement.
- No crowded same-box expression uses partial-index transforms when a safer full-group replacement is possible.
- If the scene includes axes, both axes are labeled and readable.
- If the problem specifies vertical boundaries such as `x = a` and `x = b`, both boundary lines are actually drawn and labeled.
- If a shaded graph region is shown, it is bounded by the same curve/axis/lines named in the problem statement.
- If a graph region is described as "under the curve," the shaded lower boundary is the x-axis or another stated boundary, never an accidental diagonal closure from a parametric arc.
- If readable text would require shrinking below a comfortable size, restructure the scene instead of compressing the typography.
- If the scene would leave most of the frame empty, restructure it so the answer is presented as a deliberate panel or recap instead of a tiny isolated line.
- If a multi-line derivation stack or answer card is being squeezed into a short box, treat that as a layout failure even if nothing technically overlaps.
- If a comparison table has 4 or more text-heavy columns, long sentence-style cells, or both many rows and many columns, treat that as a layout failure and split it into multiple scenes, cards, or smaller subtables.
