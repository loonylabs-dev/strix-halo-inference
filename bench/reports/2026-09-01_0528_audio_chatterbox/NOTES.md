# Why this report has a WAV and no result.json

First fenced chatterbox run, 01.09.2026. The synthesis itself SUCCEEDED —
rep1.wav here is the output — but the wrapper wrote it as a float32 WAV
(format tag 3): torchaudio's default for a float tensor, while the
wrapper's own docstring promised 16-bit PCM. bench/audiocheck.py refuses
non-integer PCM by design, except that the refusal came out of the stdlib
wave module as an unhandled exception, and audiobench died on it before
writing any result — so this directory is the failure, preserved.

Three fixes came out of it, all in the same commit as this note:
media/audio/chatterbox_tts.py saves PCM_S 16-bit explicitly,
audiocheck.load() turns the stdlib error into its own named refusal
(gate-tested against a real format-3 file now), and audiobench records an
unreadable rep as BROKEN instead of dying — a cell that fails is recorded
rather than fatal.

The clean rerun is ../2026-09-01_0530_audio_chatterbox/.
