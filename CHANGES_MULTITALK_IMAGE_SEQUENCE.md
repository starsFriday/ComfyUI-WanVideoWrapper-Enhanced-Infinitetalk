# MultiTalk Image Sequence Updates

## Overview

Support for cycling multiple input images across multi-talk frame windows is implemented via updates to both the node declaration layer and the sampler loop.

## Modified Files

1. `custom_nodes/ComfyUI-WanVideoWrapper/multitalk/nodes.py`
   - Added `WanVideoImageToVideoMultiTalkEnhanced` with an `image_count` parameter and numbered `start_image_*` inputs.
   - Preprocesses each supplied image to the expected latent resolution, stacks them, and stores as `multitalk_start_images`.
   - Publishes metadata (`multitalk_sequence_image_mode`, `multitalk_sequence_image_count`) consumed by the sampler.

2. `custom_nodes/ComfyUI-WanVideoWrapper/nodes_sampler.py`
   - Reworked the multitalk branch to read `multitalk_start_images`, fold them into a `(1, C, N, H, W)` tensor, and rotate the active conditioning image per window iteration.
   - Updated colour matching and Uni3C conditioning to pull reference frames from the active window image.
   - Falls back to zero tensors if images are missing so workflows continue to run without supplied inputs.

