# images ¶

Source: https://docs.manim.community/en/stable/reference/manim.utils.images.html

# images ¶

Image manipulation utilities.

Functions

change_to_rgba_array ( image , dtype = 'uint8' ) [source] ¶ Converts an RGB array into RGBA with the alpha value opacity maxed. Parameters : image ( RGBPixelArray ) dtype ( str ) Return type : RGBAPixelArray

drag_pixels ( frames ) [source] ¶ Parameters : frames ( Sequence [ PixelArray ] ) Return type : list[np.ndarray]

get_full_raster_image_path ( image_file_name ) [source] ¶ Parameters : image_file_name ( str | PurePath ) Return type : Path

get_full_vector_image_path ( image_file_name ) [source] ¶ Parameters : image_file_name ( str | PurePath ) Return type : Path

invert_image ( image ) [source] ¶ Parameters : image ( PixelArray ) Return type : <module ‘PIL.Image’ from ‘/home/docs/checkouts/readthedocs.org/user_builds/manimce/envs/stable/lib/python3.13/site-packages/PIL/Image.py’>
