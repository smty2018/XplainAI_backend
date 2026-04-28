# MathTex ¶

Source: https://docs.manim.community/en/stable/reference/manim.mobject.text.tex_mobject.MathTex.html

# MathTex ¶

Qualified name: manim.mobject.text.tex\_mobject.MathTex

class MathTex ( * tex_strings , arg_separator = ' ' , substrings_to_isolate = None , tex_to_color_map = None , tex_environment = 'align*' , ** kwargs ) [source] ¶ Bases: SingleStringMathTex A string compiled with LaTeX in math mode. Examples Example: Formula ¶ from manim import * class Formula ( Scene ): def construct ( self ): t = MathTex ( r "\int_a^b f'(x) dx = f(b)- f(a)" ) self . add ( t ) class Formula(Scene):
    def construct(self):
        t = MathTex(r"\int_a^b f'(x) dx = f(b)- f(a)")
        self.add(t) Notes Double-brace notation {{ ... }} can be used to split a single
string argument into multiple submobjects without having to pass
separate strings: MathTex ( r "{{ a^2 }} + {{ b^2 }} = {{ c^2 }}" ) Each {{ ... }} group and every piece of text between groups
becomes its own submobject, which is useful for TransformMatchingTex animations. For {{ to be recognised as a group opener it must appear either
at the very start of the string or be immediately preceded by a
whitespace character. {{ that follows non-whitespace — such as
in \frac{{{n}}}{k} or a^{{2}} — is left untouched, so
ordinary nested-brace LaTeX is not accidentally split.  To prevent
an unintentional split, insert a space between the two braces: {{ ... }} → { { ... } } . Tests Check that creating a MathTex works: >>> MathTex ( 'a^2 + b^2 = c^2' ) MathTex('a^2 + b^2 = c^2') Check that double brace group splitting works correctly: >>> t1 = MathTex ( '{{ a }} + {{ b }} = {{ c }}' ) >>> len ( t1 . submobjects ) 5 >>> t2 = MathTex ( r "\frac {1} {a+b\sqrt {2} }" ) >>> len ( t2 . submobjects ) 1 Methods get_part_by_tex index_of_part set_color_by_tex set_color_by_tex_to_color_map set_opacity_by_tex Sets the opacity of the tex specified. sort_alphabetically Attributes always Call a method on a mobject every frame. animate Used to animate the application of any method of self . animation_overrides color depth The depth of the mobject. fill_color If there are multiple colors (for gradient) this returns the first one font_size The font size of the tex mobject. hash_seed A unique hash representing the result of the generated mobject points. height The height of the mobject. n_points_per_curve sheen_factor stroke_color width The width of the mobject. Parameters : tex_strings ( str ) arg_separator ( str ) substrings_to_isolate ( Iterable [ str ] | None ) tex_to_color_map ( dict [ str , ParsableManimColor ] | None ) tex_environment ( str | None ) kwargs ( Any ) _break_up_by_substrings ( ) [source] ¶ Reorganize existing submobjects one layer
deeper based on the structure of tex_strings (as a list
of tex_strings) Return type : Self property _main_matches : list [ tuple [ str , str ] ] ¶ Return only the main tex_string matches. _original__init__ ( * tex_strings , arg_separator = ' ' , substrings_to_isolate = None , tex_to_color_map = None , tex_environment = 'align*' , ** kwargs ) ¶ Initialize self.  See help(type(self)) for accurate signature. Parameters : tex_strings ( str ) arg_separator ( str ) substrings_to_isolate ( Iterable [ str ] | None ) tex_to_color_map ( dict [ str , TypeAliasForwardRef ( '~manim.utils.color.core.ParsableManimColor' ) ] | None ) tex_environment ( str | None ) kwargs ( Any ) static _split_double_braces ( tex_string ) [source] ¶ Split tex_string on Manim’s {{ ... }} double-brace notation. Rules that avoid false positives on ordinary LaTeX source: {{ is only treated as a group opener when it appears at the very
start of the string or is immediately preceded by a whitespace
character.  Naturally-occurring {{ in LaTeX is usually preceded
by non-whitespace (e.g. \frac{{{n}}}{k} or a^{{2}} ), so
the whitespace guard eliminates the most common false positives
without any brace-depth bookkeeping on the outer string. Inside an open group the depth of real LaTeX braces is tracked. }} only closes the Manim group when the inner depth is zero,
so {{ a^{b^{c}} }} is handled correctly. Escape sequences are consumed as two-character units in priority
order: \\ first (escaped backslash), then \{ / \} (escaped braces).  This ensures e.g. \\}} is read as an
escaped backslash followed by a real }} rather than as \ + \} + lone } . Parameters : tex_string ( str ) Return type : list[str] property _substring_matches : list [ tuple [ str , str ] ] ¶ Return only the ‘ss’ (substring_to_isolate) matches. set_opacity_by_tex ( tex , opacity = 0.5 , remaining_opacity = None , ** kwargs ) [source] ¶ Sets the opacity of the tex specified. If ‘remaining_opacity’ is specified,
then the remaining tex will be set to that opacity. Parameters : tex ( str ) – The tex to set the opacity of. opacity ( float ) – Default 0.5. The opacity to set the tex to remaining_opacity ( float | None ) – Default None. The opacity to set the remaining tex to.
If None, then the remaining tex will not be changed kwargs ( Any ) Return type : Self
