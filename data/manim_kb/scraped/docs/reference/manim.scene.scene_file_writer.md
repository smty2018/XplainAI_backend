# scene_file_writer ¶

Source: https://docs.manim.community/en/stable/reference/manim.scene.scene_file_writer.html

# scene_file_writer ¶

The interface between scenes and ffmpeg.

Classes

SceneFileWriter | SceneFileWriter is the object that actually writes the animations played, into video files, using FFMPEG.

Functions

convert_audio ( input_path , output_path , codec_name ) [source] ¶ Parameters : input_path ( Path ) output_path ( Path | _TemporaryFileWrapper [ bytes ] ) codec_name ( str ) Return type : None

to_av_frame_rate ( fps ) [source] ¶ Parameters : fps ( float ) Return type : Fraction
