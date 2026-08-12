# Privacy and data flow

`frame2frame` is designed for local video processing. The project code has no
analytics service and does not upload frames, landmarks, poses, or rendered
videos.

## What stays local

- File and webcam frames are decoded in the local process.
- Pose inference and rendering run on the local machine.
- Face landmarks and pose observations are held in memory for processing and
  optional plotting; no face template or identity database is created.
- Output video, diagnostic plots, and synthetic demos are written only to paths
  chosen by the user.
- Audio preservation invokes a local `ffmpeg` executable.

Webcam mode writes a video when an output path is configured. Use the Python
configuration entry point with `output=None` if only a live display is intended,
and remember that other screen-recording or operating-system software is
outside this project's control.

## Network access

Normal processing can make a first-use download:

| Trigger | Download | Default destination |
|---|---|---|
| MediaPipe backend | Face Landmarker model | `~/.cache/frame2frame/face_landmarker.task` |
| Hopenet backend | published Hopenet weights | `~/.cache/frame2frame/hopenet_robust_alpha1.pkl` |
| 6DRepNet compatibility adapter | assets managed by a manually installed upstream package | upstream-defined cache |
| `scripts/fetch_examples.py` | selected Intel sample videos | `examples/inputs/` |

Project-managed model downloads are SHA-256 checked and installed atomically.
Set `FRAME2FRAME_CACHE` to redirect the model cache. To operate offline, populate
that cache in advance or pass a local model/weight path where supported.

The MediaPipe project states that media input is processed on device, while its
Tasks APIs may send performance and utilisation metrics. Review the current
[MediaPipe privacy notice](https://github.com/google-ai-edge/mediapipe#privacy)
and the policies of any optional runtime you install. Those third-party
behaviours are not controlled by `frame2frame`.

## Files to review or delete

- rendered video and plots: `output/`;
- downloaded examples: `examples/inputs/`;
- processed public examples: `examples/outputs/`;
- model cache: `~/.cache/frame2frame/` or `FRAME2FRAME_CACHE`;
- package and operating-system caches managed outside this repository.

These paths are ignored by Git. Before sharing an issue or pull request, inspect
attachments for faces, background details, audio, path names, and metadata.

## Responsible use

Head-pose data can be sensitive even though this project does not perform
identity recognition. Obtain consent before recording or analysing another
person, follow applicable workplace and local rules, and avoid using the system
for covert monitoring or high-stakes decisions. It estimates head orientation,
not attention, intent, medical state, or eye gaze.
