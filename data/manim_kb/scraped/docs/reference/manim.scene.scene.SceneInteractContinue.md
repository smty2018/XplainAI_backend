# SceneInteractContinue ¶

Source: https://docs.manim.community/en/stable/reference/manim.scene.scene.SceneInteractContinue.html

# SceneInteractContinue ¶

Qualified name: manim.scene.scene.SceneInteractContinue

class SceneInteractContinue ( sender ) [source] ¶ Bases: object Object which, when encountered in Scene.interact() , triggers
the end of the scene interaction, continuing with the rest of the
animations, if any. This object can be queued in Scene.queue for later use in Scene.interact() . Parameters : sender ( str ) sender ¶ The name of the entity which issued the end of the scene interaction,
such as "gui" or "keyboard" . Type : str Methods Attributes sender
