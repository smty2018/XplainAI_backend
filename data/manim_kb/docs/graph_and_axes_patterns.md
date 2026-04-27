# Graph and axes patterns

When a lesson benefits from a graph, use `Axes` with explicit `x_range`, `y_range`, `x_length`, and `y_length`.

Common graph workflow:

1. create axes inside a dedicated graph box
2. add axis labels
3. plot the function with `ax.plot(...)`
4. add guide lines, dots, labels, or braces as separate mobjects
5. keep text callouts outside the main curve region whenever possible

Good graph extras:

- dashed guide lines for activation points or roots
- a labeled dot for a vertex or critical point
- shaded regions for area or energy explanations
- a bottom annotation that states the visual conclusion

Avoid:

- writing long explanatory text directly over the plotted curve
- mixing formula boxes and graph boxes without enough margin
- relying on implicit default axis sizes when the scene already uses a fixed layout system
