# ChangeDecimalToValue ¶

Source: https://docs.manim.community/en/stable/reference/manim.animation.numbers.ChangeDecimalToValue.html

# ChangeDecimalToValue ¶

Qualified name: manim.animation.numbers.ChangeDecimalToValue

class ChangeDecimalToValue ( mobject = None , * args , use_override = True , ** kwargs ) [source] ¶ Bases: ChangingDecimal Animate a DecimalNumber to a target value using linear interpolation. Parameters : decimal_mob ( DecimalNumber ) – The DecimalNumber instance to animate. target_number ( int ) – The target value to transition to. kwargs ( Any ) Examples Example: ChangeDecimalToValueExample ¶ from manim import * class ChangeDecimalToValueExample ( Scene ): def construct ( self ): number = DecimalNumber ( 0 ) self . add ( number ) self . play ( ChangeDecimalToValue ( number , 10 , run_time = 3 )) self . wait () class ChangeDecimalToValueExample(Scene):
    def construct(self):
        number = DecimalNumber(0)
        self.add(number)
        self.play(ChangeDecimalToValue(number, 10, run_time=3))
        self.wait() Methods Attributes run_time _original__init__ ( decimal_mob , target_number , ** kwargs ) ¶ Initialize self.  See help(type(self)) for accurate signature. Parameters : decimal_mob ( DecimalNumber ) target_number ( int ) kwargs ( Any ) Return type : None
