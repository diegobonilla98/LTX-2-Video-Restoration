# FFHQ full restoration

This run continues the successful FFHQ aggressive adapter with the retained supervised LTX recipe. It uses ordinary flow matching, a frozen LTX-2.5 distilled base, rank-64 LoRA, BF16, batch size one, first-frame conditioning, and fully supervised degraded-to-clean target videos. It does not use the failed source bridge, repeated reference tokens, endpoint-only supervision, a GAN, or a discriminator.

Training contains JPEG compression, Gaussian blur, nearest-neighbor mosaic, bicubic low-resolution degradation, and a mixed pipeline. Every operator exposes normalized severity from 1.0 at frame one to 0.0 at the exact clean final frame. Easy, medium, hard, and extreme samples use 17, 25, 33, and 49 RGB frames. These encode to 3, 4, 5, and 7 LTX latent frames, giving 2, 3, 4, and 6 latent transitions for matching restoration-distance units.

The curriculum retains easy samples throughout training while increasing hard and extreme probability. The full run has 5,000 optimizer steps at learning rate 1.5e-5 and saves every 500 steps. All ten checkpoints are retained because later visual degradation has occurred in earlier experiments and checkpoint selection must remain possible.

Run from the project root:

```bash
./run_ffhq_full_restoration_overnight.sh
```

Monitor with:

```bash
tmux -L vtir-ffhq-full-restoration attach -t ffhq-full-restoration
tail -f logs/ffhq_full_restoration_overnight_latest.log
cat outputs/ffhq_full_restoration/run_state.json
```

The launcher is fail-closed. It builds and validates deterministic manifests, runs a real five-step optimizer smoke spanning every degradation family and all temporal shapes, verifies finite losses and changed LoRA tensors, validates the complete cache, checks an idle GPU and memory headroom, and only then starts full training. GPU-heavy phases run with swap disabled and a 90 GiB hard memory limit.
