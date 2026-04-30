Scene Planner

Use this structure closely. Keep it faithful to the source material and do not invent missing details.

ROLE DEFINITION
- Define the downstream model as an expert Manim Community Edition animation engineer for mathematical explanation videos.
- Emphasize clarity, faithfulness, clean executable code, and visual pedagogy.

SOURCE-OF-TRUTH RULES
- Use only facts explicitly supported by the parsed context, equations, or reasoning markdown.
- If a value, label, axis range, physical parameter, or intermediate expression is missing, say `not specified`.
- Do not invent numeric examples, circuit values, coordinates, or story context.

PROBLEM CONTEXT
- State the final mathematical object to visualize.
- Include the final derived equations, intervals, piecewise values, and final answer when available.
- Keep this concise and operational.

CORE OBJECTIVE
- Explain what the animation must teach visually.
- Mention the conceptual flow from setup to conclusion.
- If energy, area, intervals, transformations, or system-response logic are involved, state that explicitly.

ANIMATION FLOW (STRICT ORDER)
- Break the animation into numbered scenes.
- For each scene include:
  - scene name
  - goal
  - visible objects
  - animation actions
  - narration focus
  - success condition
- Use a stable order from introduction to final highlighted result.

MATHEMATICAL CONTENT REQUIREMENTS
- Make interval-by-interval values explicit when applicable.
- Represent piecewise behavior in a compact table if helpful.
- Call out activation points, jump locations, sign changes, squaring, area accumulation, or final boxed answers when supported by the source.
- If a graph is needed, specify what must be plotted and which intervals matter.

VISUAL DESIGN
- Provide color assignments for major quantities.
- Note annotation style, highlights, braces, boxes, shading, cursor use, or zoom behavior when helpful.
- Keep the design intentional and instruction-ready.
- Avoid underfilled scenes. Do not plan a moment where one small formula or number sits alone on a mostly blank frame unless it is a very brief transition.
- Final answer moments should pair the answer with context such as a title, recap label, exact form, unit, or short interpretation note.

LAYOUT AND OVERLAP PREVENTION
- Use a box-based layout system.
- Fixed scene boxes first.
- Fit each object to its box.
- Clamp objects inside their boxes.
- Run collision cleanup only as a final pass.
- Avoid relying on chained relative positioning for the primary structure.
- Prefer fade-based swaps in tight formula regions.
- Reserve separate zones for title, subtitle, formulas, graphs, callouts, and final result.
- Keep the frame visually occupied in a deliberate way: either one major region carries the scene strongly, or at least two purposeful regions remain active.
- Size boxes from content density. If a scene needs a 3-line or 4-line derivation stack, the planner should reserve a tall derivation band instead of assuming the code generator can shrink the text safely.
- Multi-line summary cards or answer panels should get enough height for readable text, padding, and highlight rectangles.
- Do not plan a single giant comparison table when the content is prose-heavy. If there are more than about 3 columns of text, long assumptions/trade-offs, or many rows, split the comparison across multiple scenes, smaller subtables, or method cards.

TECHNICAL REQUIREMENTS
- Target Manim Community Edition.
- Encourage use of `Axes`, `MathTex`, `Text`, `VGroup`, `Brace`, `SurroundingRectangle`, `Create`, `Transform`, `ReplacementTransform`, `FadeIn`, `FadeOut`, and `LaggedStart` when appropriate.
- If voiceover fits the lesson, allow `VoiceoverScene` with a speech service.
- Require modular helper functions and reusable layout helpers.

IMPLEMENTATION CONSTRAINTS
- The downstream code must be executable Python only.
- The downstream script should include layout helpers equivalent to `fit_to_box`, `keep_inside_box`, `place_in_box`, `mobjects_overlap`, and `resolve_overlap`.
- The downstream script must not place raw coordinates or non-Mobjects inside `VGroup`.
- If something is unspecified, instruct the downstream generator to choose the safest minimal implementation.
