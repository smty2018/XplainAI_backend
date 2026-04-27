# Voiceover and section patterns

Use `VoiceoverScene` for narrated lessons and call `self.set_speech_service(...)` before the first narration block.

Narration pattern:

```python
with self.voiceover(text="Explain the current step clearly.") as tracker:
    self.play(Write(title), run_time=min(1.2, max(0.6, tracker.duration * 0.3)))
```

Sectioning pattern:

- Use `self.next_section("scene_name")` for every major clip.
- Keep section names stable and descriptive so rerender-from-section stays predictable.
- If the scene is edited later, stable section names make partial rerender and stitching safer.

Audio pattern:

- Include `AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()`.
- Keep narration synced to the visible step rather than narrating multiple dense changes at once.
- When replacing formulas, let the narration explain the reason for the replacement before the next transformation starts.
