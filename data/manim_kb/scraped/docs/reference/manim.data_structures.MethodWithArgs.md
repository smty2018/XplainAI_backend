# MethodWithArgs ¶

Source: https://docs.manim.community/en/stable/reference/manim.data_structures.MethodWithArgs.html

# MethodWithArgs ¶

Qualified name: manim.data\_structures.MethodWithArgs

class MethodWithArgs ( method , args , kwargs ) [source] ¶ Bases: object Object containing a method which is intended to be called later
with the positional arguments args and the keyword arguments kwargs . Parameters : method ( MethodType ) args ( Iterable [ Any ] ) kwargs ( dict [ str , Any ] ) method ¶ A callable representing a method of some class. Type : MethodType args ¶ Positional arguments for method . Type : Iterable[Any] kwargs ¶ Keyword arguments for method . Type : dict[str, Any] Methods Attributes method args kwargs
