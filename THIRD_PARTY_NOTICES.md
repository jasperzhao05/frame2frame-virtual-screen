# Third-party notices

The repository does not commit model weights or full third-party source videos.
It does include two short, attributed derived media assets described below.
Optional assets are acquired at runtime and retain their upstream terms. The
project's MIT license does not replace those terms.

## Runtime components and models

### MediaPipe Face Landmarker

- Purpose: default face landmarks and canonical-face pose fit; face crop
  detector for optional deep backends.
- Software: [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe),
  Apache License 2.0.
- Model source:
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- Pinned SHA-256:
  `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- Pinned size: 3,758,596 bytes.
- Model guide:
  [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker).

The model is downloaded into the user's cache and is not redistributed here.
Review the upstream model terms and privacy notice for the intended use.

This repository redistributes the 468 vertex positions from MediaPipe's
`canonical_face_model.obj` in
`frame2frame/pose/data/canonical_face_vertices.txt`. The default adapter fits
those vertices to the detected image landmarks with the same pinhole-camera
model used by the renderer. The data remains Copyright 2020 The MediaPipe
Authors under the Apache License 2.0; its license text is included at
`LICENSES/Apache-2.0.txt`.

### Hopenet

- Purpose: optional ResNet-50 head-pose regression backend.
- Authors: Nataniel Ruiz, Eunji Chong, and James M. Rehg.
- Code/model source:
  [natanielruiz/deep-head-pose](https://github.com/natanielruiz/deep-head-pose).
- Upstream repository license: Apache License 2.0.
- Published robust-weight file ID:
  `1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR`.
- Pinned SHA-256:
  `1e0c6ddfda0e19a679607480c10875020de29b3984f187ec311c5e0802b6b6d5`.
- Pinned size: 95,924,799 bytes.

The weights are downloaded into the user's cache and are not redistributed
here. Cite the paper below when using this backend in research.

### 6DRepNet

- Purpose: experimental optional backend using a continuous 6D rotation
  representation.
- Authors: Thorsten Hempel, Ahmed A. Abdelrahman, and Ayoub Al-Hamadi.
- Package/source:
  [thohemp/6DRepNet](https://github.com/thohemp/6DRepNet), MIT License.

The upstream Python package manages its own pretrained asset and currently
declares `opencv-python`. MediaPipe requires `opencv-contrib-python`; installing
both is unsafe because the distributions own the same `cv2` namespace. This
repository therefore keeps only a compatibility adapter and does not publish a
6DRepNet installation extra. It does not pin or redistribute the upstream
asset; review its source, dependency metadata, and terms before controlled use.

### Python and media dependencies

NumPy, SciPy, OpenCV, Matplotlib, MediaPipe, PyTorch, torchvision, and optional
packages are installed separately by the Python package manager. Their own
distributions include the authoritative license texts and notices. `ffmpeg` is
an optional system executable and is not bundled.

## Example footage

`scripts/fetch_examples.py` can download selected files from the archived
[Intel IoT DevKit sample-videos repository](https://github.com/intel-iot-devkit/sample-videos).

- Licensor/source: Intel IoT DevKit `sample-videos` contributors.
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Use here: optional evaluation input; derived outputs add a virtual-screen
  overlay, pose diagnostics, and may be re-encoded.
- Storage: `examples/inputs/` and `examples/outputs/`, both Git-ignored.

Verified source digests:

| File | Bytes | SHA-256 |
|---|---:|---|
| `head-pose-face-detection-female.mp4` | 15,628,037 | `e9290821ac0e0a186e8f5cae5e3b56e8062921642d46a5f65f1ca5b311811fc5` |
| `head-pose-face-detection-male.mp4` | 15,522,596 | `994ca625f091d1422b93d97a4bd67c4de1e5ed7064c9079a68cd7526c96bfd68` |
| `head-pose-face-detection-female-and-male.mp4` | 16,788,193 | `650166430c4bf9ddc470ac17a86d1fcbd6d76c64e60ed73675fdc6b3e3d3af38` |
| `face-demographics-walking.mp4` | 6,406,124 | `91af68da819a9c0caab06c6c21414e6ebdb378ff28fee90e1f937165bc1007c6` |
| `face-demographics-walking-and-pause.mp4` | 9,406,029 | `d88ab9aa03634f66f8815db3dc940e1cdd80b098440effb20882e814fd206bf5` |
| `driver-action-recognition.mp4` | 53,804,027 | `fd24e652c92a5759127aa9c7f23fea63d73aebb4bcea72763abe72ef06a20cf0` |

When publishing a derived clip, retain this attribution, link the CC BY 4.0
license, and indicate that overlays/re-encoding modified the source.

### Derived AR use-case hero

`docs/demo-rokid-outdoor.webp` combines two attributed wearer excerpts with
separately supplied screen content:

- 02:09.800–02:12.800 of BooredAtWork / Booredatwork.com's
  [“Rokid Glasses 2025 – Next Level Augmented Reality Experience!”](https://www.youtube.com/watch?v=abE88Vve0o4),
  also available through its
  [Wikimedia Commons mirror](https://commons.wikimedia.org/wiki/File:Video_of_smart_glasses_%E2%80%93_the_Rokid_Glasses_in_2025_(with_augmented_reality).webm);
  the original review disclosed sponsorship by Rokid;
- 07:56.320–07:59.320 of Naomi 'SexyCyborg' Wu's
  [“Vuzix Blade Review—Wearable Display for a Cyberpunk Future!”](https://www.youtube.com/watch?v=sS90qEPgc50),
  used as the second wearer scene; and
- 03:32.800–03:36.320 of Game Science's
  [“Black Myth: Wukong — 13 Minutes Official Gameplay Trailer”](https://www.youtube.com/watch?v=oRLhCxC886o),
  used as separately supplied virtual-screen content.

The archived original video metadata identifies both wearer works as Creative
Commons Attribution 3.0. The repository does not assert a license to the Game
Science excerpt. None of the source material is relicensed under the project's
MIT license.

Both wearer excerpts are muted and receive one fixed square presentation crop.
The Rokid excerpt also receives one global light adjustment and a fixed blurred
fill before inference. Both are processed with the MediaPipe canonical-face
pose fit, the default FIR path, and screen distance 4. The game excerpt is
sampled as dynamic screen content and fitted inside the existing 4×2 projected
plane. Original and rendered panes are encoded side by side as a 560×280,
high-quality animated WebP with 120 presentation frames on a 20 fps timeline. Frames are
selected from the 30 fps and 25 fps wearer sources without interpolation. No
per-frame reframing, manual screen keyframes, slow motion, or reversed footage
is used. Fresh pose observations were produced for all 90 Rokid frames and all
75 Vuzix frames before presentation-rate conversion. These are per-clip
operational measurements, not claims of pose accuracy or general robustness.

The gameplay shown on the virtual screen is supplied software content; it is
not Rokid or Vuzix device output or evidence of hardware integration. This
independent project is not affiliated with or endorsed by BooredAtWork, Naomi
Wu, Game Science, Rokid, or Vuzix. Black Myth: Wukong, Rokid, Vuzix, the source
footage, subject likenesses, and associated marks are the property of their
respective rights holders and are not covered by this repository's MIT license.

### Committed derived stress example

`docs/demo-mediapipe.gif` derives from Intel IoT DevKit
[`driver-action-recognition.mp4`](https://github.com/intel-iot-devkit/sample-videos/raw/master/driver-action-recognition.mp4),
licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
It uses source time 00:00.0–00:15.0. Before inference, the source receives one
fixed 960×720 crop at `(x=120, y=0)` and is scaled to 768×576; no per-frame
reframing is used. The excerpt is muted, labeled, shown at 2x playback, and
placed beside a derived output with the project-owned virtual-screen texture
and border. The right pane was produced by the default MediaPipe backend and
FIR filter; only the screen distance is raised from `2.0` to `4.0` to separate
the plane from the face in this composition. Fresh observations occur on 422
of 450 source frames (93.8%). The longest continuous observation gap is 25
frames (0.83 seconds) and is retained to show the configured
clear-and-reacquire behavior. The final GIF contains 38 frames at 5 fps and 48
colors. This asset is a real-scene stress example, not a product use case,
backend evaluation, accuracy evidence, or a general robustness claim.

## Project-owned demo assets

`docs/demo-comparison.gif` and everything produced by `scripts/make_demo.py` or
`scripts/make_showcase.py` are generated from project-authored shapes, signals,
and textures. They contain no person footage, third-party image, or model output
and are covered by the repository's MIT License.

## BIWI-derived Screen Age research assets

`docs/screen-age-data.json` contains only aggregate, non-identifying results
computed for university research from the
[BIWI Kinect Head Pose Database](https://vision.ee.ethz.ch/datsets.html). The
repository does not contain the BIWI database, depth frames, pose files, or
subject identifiers. `docs/screen-age-scene.jpg` is the one checksum-pinned RGB
frame used by the held-out mechanism example in Panel A; the SVG embeds those
exact JPEG bytes. The projected quadrilaterals are derived from the registered
distance-4 Screen Age procedure. Panel B uses only the aggregate receipt.

The database readme makes BIWI available for non-commercial use such as
university research and education and asks users to reference the authors.
Work using the BIWI database should cite G. Fanelli, M. Dantone, J. Gall,
A. Fossati, and L. Van Gool,
“Random Forests for Real Time 3D Face Analysis,” *International Journal of
Computer Vision*, 101(3), 437–458, 2013. The BIWI data itself is not covered by
this repository's MIT License; neither is the embedded RGB frame.

## Research references

- N. Ruiz, E. Chong, and J. M. Rehg, “Fine-Grained Head Pose Estimation
  Without Keypoints,” CVPR Workshops, 2018.
  [Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w41/html/Ruiz_Fine-Grained_Head_Pose_CVPR_2018_paper.html)
- T. Hempel, A. A. Abdelrahman, and A. Al-Hamadi, “6D Rotation Representation
  for Unconstrained Head Pose Estimation,” ICIP, 2022.
  [DOI: 10.1109/ICIP46576.2022.9897219](https://doi.org/10.1109/ICIP46576.2022.9897219)
- G. Casiez, N. Roussel, and D. Vogel, “1 € Filter: A Simple Speed-based
  Low-pass Filter for Noisy Input in Interactive Systems,” CHI, 2012.
  [DOI: 10.1145/2207676.2208639](https://doi.org/10.1145/2207676.2208639)
