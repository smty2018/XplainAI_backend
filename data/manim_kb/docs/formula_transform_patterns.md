# Formula transformation patterns

For dense algebraic expressions, prefer whole-expression replacements over indexed submobject transforms.

Safer patterns:

- `ReplacementTransform(old_expr, new_expr)`
- fade out the old expression and fade in the new one in the same formula box
- split a long derivation into multiple lines or boxes

Risky patterns to avoid:

- transforming one isolated glyph from a long `MathTex` object into a distant glyph in another long `MathTex`
- keeping three long equations visible in the same box at once
- nudging crowded equations with repeated `.shift(...)`

Useful support helpers:

- `stack_in_box(...)`
- `replace_in_box(...)`
- `fade_swap(...)`

When highlighting pieces of an equation, isolate the token with `substrings_to_isolate=[...]` or use a bounding rectangle or color emphasis on the full expression rather than assuming `get_part_by_tex(...)` will always succeed.
