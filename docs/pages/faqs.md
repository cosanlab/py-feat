# FAQS

### Py-feat is detecting multiple faces in an image with a single face

This can happen sometimes particularly for videos as our current models don't use information from one frame to inform predictions for another frame. Try increasing the `face_detection_threshold` argument to `Detector.detect()` from the default of `0.5` to something like `0.8` or `0.9`. This will make the detector more conservative in what it considers a face.

### Py-feat is treating the same person as multiple identities or treating different people as the same identity

Similar to the previous issue, you can control the sensitivity of how confidently identity embeddings are retreated as different using the `face_identity_threshold` argument to `Detector.detect()` from the default of `0.8` to something higher or lower. This will make the detector more conservative or liberal respectively, in how distinct identity embeddings have to be to be considered different people.

### How can I speed things up and control memory usage?

By default all images or videos frames are processed independently in batches of size 1 using your CPU. If you have access to a CUDA-enabled GPU, you can use the `device` argument when initializing a detector instance to make use of it: `Detector(device='cuda')`. Unfortunately, macOS `'mps'` is not supported on our current model versions, but we hope to add it soon. To perform detections in parallel increase the `batch_size` argument to `Detector.detect()` from the default of 1. The largest batch size you can use without crashing your kernel is limited by the amount of VRAM available to your GPU (or RAM if you're using CPU).

In order to use batching you must either:
- use a video - where frames are all assumed to have the same dimensions
- use a list of images - where each image has the same dimensions
- use a list of images and set `output_size=(width, height)` in `.detect()` to resize all images to the same dimensions before processing

You can control parallelization of data loading using the `num_workers` argument to `.detect()`, which gets directly passed to pytorch's [DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

### Why does video processing take so long?

This was a deliberate design-tradeoff to ensure a seamless experience when processing videos of any length on any computer, regardless of memory limits or GPU availability. By default, Py-feat avoids loading an entire video into memory, only loading frames as-needed, similar to Pytorch's [unofficial video API](https://pytorch.org/vision/main/auto_examples/others/plot_video_api.html#building-a-sample-read-video-function). 

This means that you don't need to worry about your computer crashing if you're trying to process a video that doesn't fit into memory! However, it also means that theres a small latency overhead that increases with the length of the video, i.e. later frames take longer to load than earlier frames as the video needs to be "seeked" to the correct time-point.

If you already know that you have enough system memory to load the entire video at once, you can instead manually call `video_to_tensor('videofile.mp4)` from `feat.utils.io`. Then you can process the tensor by passing `data_type='tensor'` to `Detector.detect()` and proceed with batching as usual.
