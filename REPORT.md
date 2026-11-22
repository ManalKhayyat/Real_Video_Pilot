# MM-Fit real-video pilot for Mirras

## Question and scope

Can output-specific evidence checks suppress alerts that lack cue evidence while preserving repetition counts? This exploratory experiment evaluates a prototype of the proposed decision policy on real MM-Fit recordings using MediaPipe Pose Landmarker Lite. The Mirras DNN weights, deployed service, and original evaluation records were unavailable. An angle-based five-frame voting baseline approximates the reported temporal behavior. The pilot evaluates repetition counting and suppression under controlled evidence loss. It provides no trainer-validated measure of form-correction accuracy.

## Dataset and setup

- Source: MM-Fit by David Strömbäck, Sangxia Huang, and Valentin Radu. Dataset paper: https://doi.org/10.1145/3432701. RGB video release: https://doi.org/10.5281/zenodo.7672767. Set boundaries and reference repetition counts come from the annotation archive linked by https://mmfit.github.io/.
- Selection: all squat and push-up sets from the first six sequentially named workout sessions, w00–w05. Development used w00–w01 (12 clips, 121 annotated repetitions). Evaluation used w02–w05 (24 clips, 240 annotated repetitions), with 12 clips and 120 annotated repetitions per exercise. These are held-out sessions. This experiment does not assert participant independence.
- Extraction: use the labeled inclusive frame boundaries at 30 fps, downsample the RGB clips to 10 fps and a width of 640 pixels, and process each clip independently in video mode. This yielded 1,839 development and 4,248 evaluation samples. The pose model uses MediaPipe 0.10.32. Model SHA-256: 59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a.
- Geometry: use the pixel-space hip–knee–ankle angle for squats and shoulder–elbow–wrist angle for push-ups. Choose a body side using mean minimum phase-landmark quality over the first second. Keep that side fixed. Each landmark quality is the minimum of its visibility and presence values. Choose the lower and upper angle thresholds through a grid search minimizing the summed development count error of the five-frame, confidence-only, and strict policies. Freeze the thresholds before evaluating. The chosen bottom/top angles were 135°/155° for squats and 125°/145° for push-ups. Confidence threshold: 0.55.
- Alert proxy: the experimental cue is an angle threshold event during the bottom state, based on projected torso tilt of at least 35° in a squat or hip-line deviation from straight of at least 22° in a push-up. These are operational thresholds used to drive a policy comparison. The source labels provide no form or correction targets, so the angles cannot be interpreted as exercise-quality criteria.
- Interventions: process the same video-derived observations in three conditions. Use the original extracted observations for the clean condition. From 1.0 s onward, hide cue availability for four samples every two seconds in the cue-dropout condition, while leaving the phase stream and apparent cue geometry intact. In the phase-dropout condition, withhold pose observations on the same schedule. The interventions operate on the video-derived streams and do not modify the RGB pixels.

## Policies

1. **Five-frame proxy.** Convert the current phase angle to a binary state using the midpoint of the lower and upper thresholds. Use a majority vote across five valid observations, count a bottom-to-top transition, and carry state across missing observations. An active bottom state can emit one geometric alert per bottom episode.
2. **Confidence-only proxy.** Apply the same policy after rejecting observations whose phase-landmark quality is below 0.55. This proxy does not check cue-landmark quality independently.
3. **Strict phase policy.** Require two consecutive eligible samples to enter the top state, then two for the bottom and two for the return to the top. Reset on an invalid phase observation. For a cue, require two consecutive eligible bottom-phase samples, valid cue-specific evidence, and the geometric trigger. Emit at most one cue for the cycle.
4. **Short-gap phase policy.** Apply the strict policy while preserving the last established phase for a gap of up to four samples. Reset the consecutive phase and cue evidence streaks on any gap. The next supported phase observations determine further transitions.

## Results

Lower count mean absolute error (MAE) is better. Counts in the next table are totals over 24 sets with 240 reference repetitions.

| Policy | Clean MAE per set | Clean count | Phase-dropout MAE per set | Phase-dropout count | Alerts during known cue dropout |
|---|---:|---:|---:|---:|---:|
| Five-frame proxy | 3.71 | 165 | 4.54 | 135 | 5 |
| Confidence-only proxy | 3.71 | 165 | 4.54 | 135 | 5 |
| Strict phase | 5.08 | 118 | 9.33 | 16 | 0 |
| Short-gap phase | 5.08 | 118 | 5.92 | 98 | 0 |

Clean count MAE for the five-frame proxy was 3.00 for squats and 4.42 for push-ups. The two phase policies reached 3.25 and 6.92, respectively. Under clean input the five-frame proxy had 82 net within-set shortfall units and 7 excess units, while both phase policies had 122 shortfall units and no excess. These units come from count differences within labeled sets. They do not identify individual missed repetitions because per-repetition timings were unavailable.

Under cue-only dropout the repetition estimates were identical to clean estimates for every clip and policy. Five of the 19 proxy alerts occurred inside a known cue dropout. No phase-policy alert occurred inside a known cue dropout, and these policies emitted three total alerts each in that condition. The clean policies emitted 19 proxy alerts and four phase-policy alerts. All alerts in all conditions occurred on push-up clips. Suppression is an expected consequence of applying a hard validity gate to an imposed missing-evidence flag. It does not establish better clinical advice, correct recognition of real-world occlusion, or retention of warranted cues.

Under phase dropout, the short-gap policy recovered 82 counted cycles relative to the strict policy and reduced MAE from 9.33 to 5.92. This still exceeded the five-frame proxy's 4.54 MAE. The phase gate accepted all original phase samples at the selected confidence threshold, so the confidence-only proxy had the same counts as the five-frame proxy. The dataset uses a restricted recording setup. Results are descriptive for these sessions, and the calibration selected counting thresholds jointly across policies.

## Manuscript-ready results paragraph

We conducted an exploratory comparison on 24 real MM-Fit RGB exercise sets from four held-out workout sessions, comprising 12 squat sets and 12 push-up sets with 240 annotated repetitions. All policies received the same MediaPipe pose streams, and angle thresholds were selected using 12 sets from two separate development sessions. On unmodified streams, the five-frame voting proxy obtained a mean absolute count error of 3.71 repetitions per set, compared with 5.08 for the strict phase policy. Under controlled pose dropout, a four-sample recovery window reduced phase-policy count error from 9.33 to 5.92 repetitions per set, while the voting proxy achieved 4.54. Under a separate cue-evidence dropout intervention, the strict and short-gap policies emitted no geometric alerts during known unavailable-evidence windows, compared with five for the voting proxy. These results demonstrate a trade-off between evidence conservatism and count completeness in the tested policies. The comparison used an angle-based proxy for the deployed classifier and did not validate correction accuracy against expert annotations.

## Reproduce

The archive contains the clip manifest, frozen thresholds, evaluation outputs, source scripts, and the four derived, non-image policy input streams per sample. Recompute all aggregate and per-clip results with Python 3.12 and NumPy 1.26.4 by running `python evaluate.py` from the extracted folder. For the full video-to-pose path, install MediaPipe 0.10.32 and OpenCV 4.11.0.86. Use FFmpeg and FFprobe on PATH. Download the model to `data/pose_landmarker_lite.task` from https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task, then run `python mmfit_annotations.py`, `python get_clips.py`, and `python extract.py`. To rebuild the policy inputs from those pose arrays, move the supplied `features` folder out of the extracted folder before rerunning `evaluate.py`. The archive contains no RGB clips or model weights. Use the published dataset and its terms for originals.
