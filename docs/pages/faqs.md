# FAQs, Tips & Known Issues

Here are a few general guidelines and common questions when using `py-feat`. We are always actively trying to improve the toolbox and make it easier to use — please help us out by contributing on GitHub!

```{note}
**Always spot-check your detections!** While we've done our best to thoroughly test, benchmark, and document all the [pre-trained models](models.md) included in Py-Feat, it's always possible that real-world images and videos reveal a quirk or limitation of an otherwise high-performing detector.
```

## Common questions

### Py-feat is detecting multiple faces in an image with a single face

This can happen sometimes, particularly for videos, as our current models don't use information from one frame to inform predictions for another frame. Try increasing the `face_detection_threshold` argument to `Detector.detect()` from the default of `0.5` to something like `0.8` or `0.9`. This will make the detector more conservative in what it considers a face. You can also switch to the more accurate detector with `Detector(face_model='retinaface')` (88.9% vs 55.5% WIDERFACE-Hard AP), or use `Detectorv2`, which also uses RetinaFace.

### In what order are detected faces returned?

Within a frame, faces are returned in the order the face detector emits them — this order is **not** guaranteed to be stable across frames, and a given person will not necessarily occupy the same row position from frame to frame. To group detections by person across a video, use the identity labels (`Fex.identities`, computed via `Fex.compute_identities()`) rather than row position. If you need a deterministic per-frame order, sort each frame's rows yourself (e.g. by `FaceRectX`).

### Py-feat is treating the same person as multiple identities or treating different people as the same identity

Similar to the previous issue, you can control the sensitivity of how confidently identity embeddings are retreated as different using the `face_identity_threshold` argument to `Detector.detect()` from the default of `0.8` to something higher or lower. This will make the detector more conservative or liberal respectively, in how distinct identity embeddings have to be to be considered different people.

### How can I speed things up and control memory usage?

By default all images or video frames are processed independently in batches of size 1 using your CPU. If you have a GPU, use the `device` argument when initializing a detector to make use of it — `Detector(device='cuda')` for NVIDIA GPUs or `Detector(device='mps')` for Apple Silicon. Both `Detector` and `Detectorv2` support CUDA and MPS. To perform detections in parallel, increase the `batch_size` argument to `Detector.detect()` from the default of 1. The largest batch size you can use without crashing your kernel is limited by the amount of VRAM available to your GPU (or RAM if you're using CPU).

In order to use batching you must either:
- use a video - where frames are all assumed to have the same dimensions
- use a list of images - where each image has the same dimensions
- use a list of images and set `output_size=(width, height)` in `.detect()` to resize all images to the same dimensions before processing

You can control parallelization of data loading using the `num_workers` argument to `.detect()`, which gets directly passed to pytorch's [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). Note that on Apple Silicon `num_workers > 0` is frequently *slower* than the default of `0`, so we recommend leaving it at `0` unless you've benchmarked a speedup on your own hardware.

### Why does video processing take so long?

This was a deliberate design-tradeoff to ensure a seamless experience when processing videos of any length on any computer, regardless of memory limits or GPU availability. By default, Py-feat avoids loading an entire video into memory, only loading frames as-needed, similar to Pytorch's [unofficial video API](https://pytorch.org/vision/main/auto_examples/others/plot_video_api.html#building-a-sample-read-video-function). 

This means that you don't need to worry about your computer crashing if you're trying to process a video that doesn't fit into memory! However, it also means that there's a small latency overhead that increases with the length of the video, i.e. later frames take longer to load than earlier frames as the video needs to be "seeked" to the correct time-point.

If you already know that you have enough system memory to load the entire video at once, you can instead manually call `video_to_tensor('videofile.mp4')` from `feat.utils.io`. Then you can process the tensor by passing `data_type='tensor'` to `Detector.detect()` and proceed with batching as usual.

## Known issues

- Detectors can be sensitive to differences in image sizes, such that very large or very small images result in very different predictions. This is largely due to differences in the hyper-parameters and image sizes used to train the underlying pre-trained models. To partially help with this issue, since `0.5.0` Py-Feat supports passing keyword arguments to underlying models during initialization or detection, e.g. `detector = Detector(facepose_model_kwargs={'keep_top_k': 500})`. You can follow [this issue](https://github.com/cosanlab/py-feat/issues/135) for more details.

## Community & getting help

- [Discourse Community](https://www.askpbs.org/c/py-feat/26): a Stack Overflow-like forum where you can view, contribute, and vote on questions regarding `py-feat` usage. Please ask questions here first so other users can benefit from the answers!
- [Open a GitHub issue](https://github.com/cosanlab/py-feat/issues) for all code-related problems. You can also do so by clicking the GitHub icon at the top of any page.
