# Example: FixedInFrameMObjectTest

Category: Special Camera Settings
Source: https://docs.manim.community/en/stable/examples.html

References: ThreeDScene set_camera_orientation() add_fixed_in_frame_mobjects()

```python
class FixedInFrameMObjectTest(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        text3d = Text("This is a 3D text")
        self.add_fixed_in_frame_mobjects(text3d)
        text3d.to_corner(UL)
        self.add(axes)
        self.wait()
```
