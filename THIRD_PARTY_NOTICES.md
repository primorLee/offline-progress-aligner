# Third-party notices

## HOST SmoothDTW (training only)

`offline_progress_aligner/vendor/host_smooth_dtw.py` is the unmodified SmoothDTW implementation used by the original training run. It is attributed to the [HOST repository](https://github.com/CGuangyan-BIT/HOST), pinned source:

https://github.com/CGuangyan-BIT/HOST/blob/9f3bba57792aa5053ac600b1b7a4625f96ea6662/alignment/tcc/smooth_dtw.py

The included [HOST-MIT.txt](offline_progress_aligner/vendor/HOST-MIT.txt) preserves its license verbatim, including the original copyright holder, **The FastWAM Authors**. The corresponding license source is:

https://github.com/CGuangyan-BIT/HOST/blob/9f3bba57792aa5053ac600b1b7a4625f96ea6662/policy_training/LICENSE

Its file header credits D2TW (Hadji et al., 2021) and the original TensorFlow reference. That header is retained. HOST is an upstream implementation used by the loss, not the source of this release's trained adapter weights. Inference uses the local hard-DTW implementation and does not import SmoothDTW.

## Datasets and separate feature extractor

The adapter's training used [H&R / Human2Robot](https://huggingface.co/datasets/dannyXSC/HumanAndRobot) and [RH20T](https://rh20t.github.io/). Dataset assets are not included; obtain them from their publishers under their applicable terms. RH20T-NC also restricts commercial model use, reflected in the pretrained-weight license.

The frozen upstream interaction-token extractor is a separate component and is not redistributed. Its feature identity is recorded in the model metadata. This package does not claim to contain that encoder, raw-video feature extraction, the original HOST visual backbone, a world model, or a control policy.

NumPy, PyTorch and safetensors remain under their respective upstream licenses; they are package dependencies rather than vendored copies.
