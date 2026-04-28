# typing ¶

Source: https://docs.manim.community/en/stable/reference/manim.typing.html

# typing ¶

Custom type definitions used in Manim.

Note for developers

Around the source code there are multiple strings which look like this:

```
'''


[CATEGORY]


<category_name>


'''
```

All type aliases defined under those strings will be automatically
classified under that category.

If you need to define a new category, respect the format described above.

Type Aliases

### Primitive data types ¶

class ManimFloat ¶ np . float64 A double-precision floating-point value (64 bits, or 8 bytes),
according to the IEEE 754 standard.

class ManimInt ¶ np . int64 A long integer (64 bits, or 8 bytes). It can take values between \(-2^{63}\) and \(+2^{63} - 1\) ,
which expressed in base 10 is a range between around \(-9.223 \cdot 10^{18}\) and \(+9.223 \cdot 10^{18}\) .

### Color types ¶

class ManimColorDType ¶ ManimFloat Data type used in ManimColorInternal : a
double-precision float between 0 and 1.

class FloatRGB ¶ NDArray[ ManimColorDType ] shape: (3,) A numpy.ndarray of 3 floats between 0 and 1, representing a
color in RGB format. Its components describe, in order, the intensity of Red, Green, and
Blue in the represented color.

class FloatRGBLike ¶ FloatRGB | tuple[float, float, float] shape: (3,) An array of 3 floats between 0 and 1, representing a color in RGB
format. This represents anything which can be converted to a FloatRGB NumPy
array.

class FloatRGB_Array ¶ NDArray[ ManimColorDType ] shape: (M, 3) A numpy.ndarray of many rows of 3 floats representing RGB colors.

class FloatRGBLike_Array ¶ FloatRGB_Array | Sequence[ FloatRGBLike ] shape: (M, 3) An array of many rows of 3 floats representing RGB colors. This represents anything which can be converted to a FloatRGB_Array NumPy
array.

class IntRGB ¶ NDArray[ ManimInt ] shape: (3,) A numpy.ndarray of 3 integers between 0 and 255,
representing a color in RGB format. Its components describe, in order, the intensity of Red, Green, and
Blue in the represented color.

class IntRGBLike ¶ IntRGB | tuple[int, int, int] shape: (3,) An array of 3 integers between 0 and 255, representing a color in RGB
format. This represents anything which can be converted to an IntRGB NumPy
array.

class FloatRGBA ¶ NDArray[ ManimColorDType ] shape: (4,) A numpy.ndarray of 4 floats between 0 and 1, representing a
color in RGBA format. Its components describe, in order, the intensity of Red, Green, Blue
and Alpha (opacity) in the represented color.

class FloatRGBALike ¶ FloatRGBA | tuple[float, float, float, float] shape: (4,) An array of 4 floats between 0 and 1, representing a color in RGBA
format. This represents anything which can be converted to a FloatRGBA NumPy
array.

class FloatRGBA_Array ¶ NDArray[ ManimColorDType ] shape: (M, 4) A numpy.ndarray of many rows of 4 floats representing RGBA colors.

class FloatRGBALike_Array ¶ FloatRGBA_Array | Sequence[ FloatRGBALike ] shape: (M, 4) An array of many rows of 4 floats representing RGBA colors. This represents anything which can be converted to a FloatRGBA_Array NumPy
array.

class IntRGBA ¶ NDArray[ ManimInt ] shape: (4,) A numpy.ndarray of 4 integers between 0 and 255,
representing a color in RGBA format. Its components describe, in order, the intensity of Red, Green, Blue
and Alpha (opacity) in the represented color.

class IntRGBALike ¶ IntRGBA | tuple[int, int, int, int] shape: (4,) An array of 4 integers between 0 and 255, representing a color in RGBA
format. This represents anything which can be converted to an IntRGBA NumPy
array.

class FloatHSV ¶ FloatRGB shape: (3,) A numpy.ndarray of 3 floats between 0 and 1, representing a
color in HSV (or HSB) format. Its components describe, in order, the Hue, Saturation and Value (or
Brightness) in the represented color.

class FloatHSVLike ¶ FloatRGBLike shape: (3,) An array of 3 floats between 0 and 1, representing a color in HSV (or
HSB) format. This represents anything which can be converted to a FloatHSV NumPy
array.

class FloatHSVA ¶ FloatRGBA shape: (4,) A numpy.ndarray of 4 floats between 0 and 1, representing a
color in HSVA (or HSBA) format. Its components describe, in order, the Hue, Saturation and Value (or
Brightness) in the represented color.

class FloatHSVALike ¶ FloatRGBALike shape: (4,) An array of 4 floats between 0 and 1, representing a color in HSVA (or
HSBA) format. This represents anything which can be converted to a FloatHSVA NumPy
array.

class FloatHSL ¶ FloatRGB shape: (3,) A numpy.ndarray of 3 floats between 0 and 1, representing a
color in HSL format. Its components describe, in order, the Hue, Saturation and Lightness
in the represented color.

class FloatHSLLike ¶ FloatRGBLike shape: (3,) An array of 3 floats between 0 and 1, representing a color in HSL format. This represents anything which can be converted to a FloatHSL NumPy
array.

class ManimColorInternal ¶ FloatRGBA shape: (4,) Internal color representation used by ManimColor ,
following the RGBA format. It is a numpy.ndarray consisting of 4 floats between 0 and
1, describing respectively the intensities of Red, Green, Blue and
Alpha (opacity) in the represented color.

### Point types ¶

class PointDType ¶ ManimFloat Default type for arrays representing points: a double-precision
floating point value.

class Point2D ¶ NDArray[ PointDType ] shape: (2,) A NumPy array representing a 2-dimensional point: [float, float] .

class Point2DLike ¶ Point2D | tuple[float, float] shape: (2,) A 2-dimensional point: [float, float] . This represents anything which can be converted to a Point2D NumPy
array.

class Point2D_Array ¶ NDArray[ PointDType ] shape: (M, 2) A NumPy array representing a sequence of Point2D objects: [[float, float], ...] .

class Point2DLike_Array ¶ Point2D_Array | Sequence[ Point2DLike ] shape: (M, 2) An array of Point2DLike objects: [[float, float], ...] . This represents anything which can be converted to a Point2D_Array NumPy array. Please refer to the documentation of the function you are using for
further type information.

class Point3D ¶ NDArray[ PointDType ] shape: (3,) A NumPy array representing a 3-dimensional point: [float, float, float] .

class Point3DLike ¶ Point3D | tuple[float, float, float] shape: (3,) A 3-dimensional point: [float, float, float] . This represents anything which can be converted to a Point3D NumPy
array.

class Point3D_Array ¶ NDArray[ PointDType ] shape: (M, 3) A NumPy array representing a sequence of Point3D objects: [[float, float, float], ...] .

class Point3DLike_Array ¶ Point3D_Array | Sequence[ Point3DLike ] shape: (M, 3) An array of Point3DLike objects: [[float, float, float], ...] . This represents anything which can be converted to a Point3D_Array NumPy array. Please refer to the documentation of the function you are using for
further type information.

class PointND ¶ NDArray[ PointDType ] shape: (N,) A NumPy array representing an N-dimensional point: [float, ...] .

class PointNDLike ¶ PointND | Sequence[float] shape: (N,) An N-dimensional point: [float, ...] . This represents anything which can be converted to a PointND NumPy
array.

class PointND_Array ¶ NDArray[ PointDType ] shape: (M, N) A NumPy array representing a sequence of PointND objects: [[float, ...], ...] .

class PointNDLike_Array ¶ PointND_Array | Sequence[ PointNDLike ] shape: (M, N) An array of PointNDLike objects: [[float, ...], ...] . This represents anything which can be converted to a PointND_Array NumPy array. Please refer to the documentation of the function you are using for
further type information.

### Vector types ¶

class Vector2D ¶ NDArray[ PointDType ] shape: (2,) A NumPy array representing a 2-dimensional vector: [float, float] . Caution Do not confuse with the Vector or Arrow VMobjects!

class Vector2DLike ¶ NDArray[ PointDType ] | tuple[float, float] shape: (2,) A 2-dimensional vector: [float, float] . This represents anything which can be converted to a Vector2D NumPy
array. Caution Do not confuse with the Vector or Arrow VMobjects!

class Vector2D_Array ¶ NDArray[ PointDType ] shape: (M, 2) A NumPy array representing a sequence of Vector2D objects: [[float, float], ...] .

class Vector2DLike_Array ¶ Vector2D_Array | Sequence[ Vector2DLike ] shape: (M, 2) An array of Vector2DLike objects: [[float, float], ...] . This represents anything which can be converted to a Vector2D_Array NumPy array.

class Vector3D ¶ NDArray[ PointDType ] shape: (3,) A NumPy array representing a 3-dimensional vector: [float, float, float] . Caution Do not confuse with the Vector or Arrow3D VMobjects!

class Vector3DLike ¶ NDArray[ PointDType ] | tuple[float, float, float] shape: (3,) A 3-dimensional vector: [float, float, float] . This represents anything which can be converted to a Vector3D NumPy
array. Caution Do not confuse with the Vector or Arrow3D VMobjects!

class Vector3D_Array ¶ NDArray[ PointDType ] shape: (M, 3) An NumPy array representing a sequence of Vector3D objects: [[float, float, float], ...] .

class Vector3DLike_Array ¶ NDArray[ PointDType ] | Sequence[ Vector3DLike ] shape: (M, 3) An array of Vector3DLike objects: [[float, float, float], ...] . This represents anything which can be converted to a Vector3D_Array NumPy array.

class VectorND ¶ NDArray[ PointDType ] shape (N,) A NumPy array representing an \(N\) -dimensional vector: [float, ...] . Caution Do not confuse with the Vector VMobject! This type alias
is named “VectorND” instead of “Vector” to avoid potential name
collisions.

class VectorNDLike ¶ NDArray[ PointDType ] | Sequence[float] shape (N,) An \(N\) -dimensional vector: [float, ...] . This represents anything which can be converted to a VectorND NumPy
array. Caution Do not confuse with the Vector VMobject! This type alias
is named “VectorND” instead of “Vector” to avoid potential name
collisions.

class VectorND_Array ¶ NDArray[ PointDType ] shape (M, N) A NumPy array representing a sequence of VectorND objects: [[float, ...], ...] .

class VectorNDLike_Array ¶ NDArray[ PointDType ] | Sequence[ VectorNDLike ] shape (M, N) An array of VectorNDLike objects: [[float, ...], ...] . This represents anything which can be converted to a VectorND_Array NumPy array.

class RowVector ¶ NDArray[ PointDType ] shape: (1, N) A row vector: [[float, ...]] .

class ColVector ¶ NDArray[ PointDType ] shape: (N, 1) A column vector: [[float], [float], ...] .

### Matrix types ¶

class MatrixMN ¶ NDArray[ PointDType ] shape: (M, N) A matrix: [[float, ...], [float, ...], ...] .

class Zeros ¶ MatrixMN shape: (M, N) A MatrixMN filled with zeros, typically created with numpy.zeros((M, N)) .

### Bézier types ¶

class QuadraticBezierPoints ¶ Point3D_Array shape: (3, 3) A Point3D_Array of three 3D control points for a single quadratic Bézier
curve: [[float, float, float], [float, float, float], [float, float, float]] .

class QuadraticBezierPointsLike ¶ QuadraticBezierPoints | tuple[ Point3DLike , Point3DLike , Point3DLike ] shape: (3, 3) A Point3DLike_Array of three 3D control points for a single quadratic Bézier
curve: [[float, float, float], [float, float, float], [float, float, float]] . This represents anything which can be converted to a QuadraticBezierPoints NumPy array.

class QuadraticBezierPoints_Array ¶ NDArray[ PointDType ] shape: (N, 3, 3) A NumPy array containing \(N\) QuadraticBezierPoints objects: [[[float, float, float], [float, float, float], [float, float, float]], ...] .

class QuadraticBezierPointsLike_Array ¶ QuadraticBezierPoints_Array | Sequence[ QuadraticBezierPointsLike ] shape: (N, 3, 3) A sequence of \(N\) QuadraticBezierPointsLike objects: [[[float, float, float], [float, float, float], [float, float, float]], ...] . This represents anything which can be converted to a QuadraticBezierPoints_Array NumPy array.

class QuadraticBezierPath ¶ Point3D_Array shape: (3*N, 3) A Point3D_Array of \(3N\) points, where each one of the \(N\) consecutive blocks of 3 points represents a quadratic
Bézier curve: [[float, float, float], ...], ...] . Please refer to the documentation of the function you are using for
further type information.

class QuadraticBezierPathLike ¶ Point3DLike_Array shape: (3*N, 3) A Point3DLike_Array of \(3N\) points, where each one of the \(N\) consecutive blocks of 3 points represents a quadratic
Bézier curve: [[float, float, float], ...], ...] . This represents anything which can be converted to a QuadraticBezierPath NumPy array. Please refer to the documentation of the function you are using for
further type information.

class QuadraticSpline ¶ QuadraticBezierPath shape: (3*N, 3) A special case of QuadraticBezierPath where all the \(N\) quadratic Bézier curves are connected, forming a quadratic spline: [[float, float, float], ...], ...] . Please refer to the documentation of the function you are using for
further type information.

class QuadraticSplineLike ¶ QuadraticBezierPathLike shape: (3*N, 3) A special case of QuadraticBezierPathLike where all the \(N\) quadratic Bézier curves are connected, forming a quadratic spline: [[float, float, float], ...], ...] . This represents anything which can be converted to a QuadraticSpline NumPy array. Please refer to the documentation of the function you are using for
further type information.

class CubicBezierPoints ¶ Point3D_Array shape: (4, 3) A Point3D_Array of four 3D control points for a single cubic Bézier curve: [[float, float, float], [float, float, float], [float, float, float], [float, float, float]] .

class CubicBezierPointsLike ¶ CubicBezierPoints | tuple[ Point3DLike , Point3DLike , Point3DLike , Point3DLike ] shape: (4, 3) A Point3DLike_Array of 4 control points for a single cubic Bézier curve: [[float, float, float], [float, float, float], [float, float, float], [float, float, float]] . This represents anything which can be converted to a CubicBezierPoints NumPy array.

class CubicBezierPoints_Array ¶ NDArray[ PointDType ] shape: (N, 4, 3) A NumPy array containing \(N\) CubicBezierPoints objects: [[[float, float, float], [float, float, float], [float, float, float], [float, float, float]], ...] .

class CubicBezierPointsLike_Array ¶ CubicBezierPoints_Array | Sequence[ CubicBezierPointsLike ] shape: (N, 4, 3) A sequence of \(N\) CubicBezierPointsLike objects: [[[float, float, float], [float, float, float], [float, float, float], [float, float, float]], ...] . This represents anything which can be converted to a CubicBezierPoints_Array NumPy array.

class CubicBezierPath ¶ Point3D_Array shape: (4*N, 3) A Point3D_Array of \(4N\) points, where each one of the \(N\) consecutive blocks of 4 points represents a cubic Bézier
curve: [[float, float, float], ...], ...] . Please refer to the documentation of the function you are using for
further type information.

class CubicBezierPathLike ¶ Point3DLike_Array shape: (4*N, 3) A Point3DLike_Array of \(4N\) points, where each one of the \(N\) consecutive blocks of 4 points represents a cubic Bézier
curve: [[float, float, float], ...], ...] . This represents anything which can be converted to a CubicBezierPath NumPy array. Please refer to the documentation of the function you are using for
further type information.

class CubicSpline ¶ CubicBezierPath shape: (4*N, 3) A special case of CubicBezierPath where all the \(N\) cubic
Bézier curves are connected, forming a quadratic spline: [[float, float, float], ...], ...] . Please refer to the documentation of the function you are using for
further type information.

class CubicSplineLike ¶ CubicBezierPathLike shape: (4*N, 3) A special case of CubicBezierPath where all the \(N\) cubic
Bézier curves are connected, forming a quadratic spline: [[float, float, float], ...], ...] . This represents anything which can be converted to a CubicSpline NumPy array. Please refer to the documentation of the function you are using for
further type information.

class BezierPoints ¶ Point3D_Array shape: (PPC, 3) A Point3D_Array of \(\text{PPC}\) control points
( \(\text{PPC: Points Per Curve} = n + 1\) ) for a single \(n\) -th degree Bézier curve: [[float, float, float], ...] . Please refer to the documentation of the function you are using for
further type information.

class BezierPointsLike ¶ Point3DLike_Array shape: (PPC, 3) A Point3DLike_Array of \(\text{PPC}\) control points
( \(\text{PPC: Points Per Curve} = n + 1\) ) for a single \(n\) -th degree Bézier curve: [[float, float, float], ...] . This represents anything which can be converted to a BezierPoints NumPy array. Please refer to the documentation of the function you are using for
further type information.

class BezierPoints_Array ¶ NDArray[ PointDType ] shape: (N, PPC, 3) A NumPy array of \(N\) BezierPoints objects containing \(\text{PPC}\) Point3D objects each
( \(\text{PPC: Points Per Curve} = n + 1\) ): [[[float, float, float], ...], ...] . Please refer to the documentation of the function you are using for
further type information.

class BezierPointsLike_Array ¶ BezierPoints_Array | Sequence[ BezierPointsLike ] shape: (N, PPC, 3) A sequence of \(N\) BezierPointsLike objects containing \(\text{PPC}\) Point3DLike objects each
( \(\text{PPC: Points Per Curve} = n + 1\) ): [[[float, float, float], ...], ...] . This represents anything which can be converted to a BezierPoints_Array NumPy array. Please refer to the documentation of the function you are using for
further type information.

class BezierPath ¶ Point3D_Array shape: (PPC*N, 3) A Point3D_Array of \(\text{PPC} \cdot N\) points, where each
one of the \(N\) consecutive blocks of \(\text{PPC}\) control
points ( \(\text{PPC: Points Per Curve} = n + 1\) ) represents a
Bézier curve of \(n\) -th degree: [[float, float, float], ...], ...] . Please refer to the documentation of the function you are using for
further type information.

class BezierPathLike ¶ Point3DLike_Array shape: (PPC*N, 3) A Point3DLike_Array of \(\text{PPC} \cdot N\) points, where each
one of the \(N\) consecutive blocks of \(\text{PPC}\) control
points ( \(\text{PPC: Points Per Curve} = n + 1\) ) represents a
Bézier curve of \(n\) -th degree: [[float, float, float], ...], ...] . This represents anything which can be converted to a BezierPath NumPy array. Please refer to the documentation of the function you are using for
further type information.

class Spline ¶ BezierPath shape: (PPC*N, 3) A special case of BezierPath where all the \(N\) Bézier curves
consisting of \(\text{PPC}\) Point3D objects
( \(\text{PPC: Points Per Curve} = n + 1\) ) are connected, forming
an \(n\) -th degree spline: [[float, float, float], ...], ...] . Please refer to the documentation of the function you are using for
further type information.

class SplineLike ¶ BezierPathLike shape: (PPC*N, 3) A special case of BezierPathLike where all the \(N\) Bézier curves
consisting of \(\text{PPC}\) Point3D objects
( \(\text{PPC: Points Per Curve} = n + 1\) ) are connected, forming
an \(n\) -th degree spline: [[float, float, float], ...], ...] . This represents anything which can be converted to a Spline NumPy array. Please refer to the documentation of the function you are using for
further type information.

class FlatBezierPoints ¶ NDArray[ PointDType ] | tuple[float, ...] shape: (3*PPC*N,) A flattened array of Bézier control points: [float, ...] .

### Function types ¶

class FunctionOverride ¶ Callable Function type returning an Animation for the specified Mobject .

class PathFuncType ¶ Callable[[ Point3DLike , Point3DLike , float], Point3DLike ] Function mapping two Point3D objects and an alpha value to a new Point3D .

class MappingFunction ¶ Callable[[ Point3D ], Point3D ] A function mapping a Point3D to another Point3D .

class MultiMappingFunction ¶ Callable[[ Point3D_Array ], Point3D_Array ] A function mapping a Point3D_Array to another Point3D_Array .

### Image types ¶

class PixelArray ¶ NDArray[ ManimInt ] shape: (height, width) | (height, width, 3) | (height, width, 4) A rasterized image with a height of height pixels and a width of width pixels. Every value in the array is an integer from 0 to 255. Every pixel is represented either by a single integer indicating its
lightness (for greyscale images), an RGB_Array_Int or an RGBA_Array_Int .

class GrayscalePixelArray ¶ PixelArray shape: (height, width) A 100% opaque grayscale PixelArray , where every pixel value is a ManimInt indicating its lightness (black -> gray -> white).

class RGBPixelArray ¶ PixelArray shape: (height, width, 3) A 100% opaque PixelArray in color, where every pixel value is an RGB_Array_Int object.

class RGBAPixelArray ¶ PixelArray shape: (height, width, 4) A PixelArray in color where pixels can be transparent. Every pixel
value is an RGBA_Array_Int object.

### Path types ¶

class StrPath ¶ str | PathLike [ str ] A string or os.PathLike representing a path to a
directory or file.

class StrOrBytesPath ¶ str | bytes | PathLike [ str ] | PathLike [ bytes ] A string, bytes or os.PathLike object representing a path
to a directory or file.
