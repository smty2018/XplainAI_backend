# Prism ¶

Source: https://docs.manim.community/en/stable/reference/manim.mobject.three_d.three_dimensions.Prism.html

# Prism ¶

Qualified name: manim.mobject.three\_d.three\_dimensions.Prism

class Prism ( dimensions = [3, 2, 1] , ** kwargs ) [source] ¶ Bases: Cube A right rectangular prism (or rectangular cuboid).
Defined by the length of each side in [x, y, z] format. Parameters : dimensions ( Vector3DLike ) – Dimensions of the Prism in [x, y, z] format. kwargs ( Any ) Examples Example: ExamplePrism ¶ from manim import * class ExamplePrism ( ThreeDScene ): def construct ( self ): self . set_camera_orientation ( phi = 60 * DEGREES , theta = 150 * DEGREES ) prismSmall = Prism ( dimensions = [ 1 , 2 , 3 ]) . rotate ( PI / 2 ) prismLarge = Prism ( dimensions = [ 1.5 , 3 , 4.5 ]) . move_to ([ 2 , 0 , 0 ]) self . add ( prismSmall , prismLarge ) class ExamplePrism(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=150 * DEGREES)
        prismSmall = Prism(dimensions=[1, 2, 3]).rotate(PI / 2)
        prismLarge = Prism(dimensions=[1.5, 3, 4.5]).move_to([2, 0, 0])
        self.add(prismSmall, prismLarge) Methods generate_points Creates the sides of the Prism . Attributes always Call a method on a mobject every frame. animate Used to animate the application of any method of self . animation_overrides color depth The depth of the mobject. fill_color If there are multiple colors (for gradient) this returns the first one height The height of the mobject. n_points_per_curve sheen_factor stroke_color width The width of the mobject. _original__init__ ( dimensions = [3, 2, 1] , ** kwargs ) ¶ Initialize self.  See help(type(self)) for accurate signature. Parameters : dimensions ( Vector3DLike ) kwargs ( Any ) Return type : None generate_points ( ) [source] ¶ Creates the sides of the Prism . Return type : None
