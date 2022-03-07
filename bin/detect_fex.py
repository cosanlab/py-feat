#!/usr/bin/env python3
"""
Multithreaded emotion & facial features detector from videos.

Usage:
    python3 detect_fex.py
    Detect facial landmarks, emotions, and action units from video file or see a demo with webcam.

Arguments:
    inputFname: str
            Path to video file to process. If no video file is passed, then provide demo with camera.
    ouputFname: str
            Path to output file to write to.
    skip_frames: int, default = 1
            Process every n-th frame for faster speed. 
    n_jobs: int, default=1
            The number of jobs to use for the computation. Default uses 1 core. 
            -1 means using all processors. 

Outputs:
    If outputFname is specified, emotion predictions, face and landmark detectionresults will be saved as a csv file.
Keyboard shortcuts:
    ESC - exit
"""

if __name__ == "__main__":
    import sys
    import argparse
    import os

    print(__doc__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputFname", help="str, Path to video.", type=str, default=0
    )
    parser.add_argument(
        "-o", "--outputFname", help="str, Path to output file.", type=str
    )
    parser.add_argument(
        "-s",
        "--skip_frames",
        help="int, Process every n-th frame.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-n", "--n_jobs", help="int, Number of processors to use", type=int, default=1
    )
    parser.add_argument(
        "-v", "--verbose", help="int, Number of processors to use", default=False
    )
    args = parser.parse_args()

    if not args.outputFname:
        if args.inputFname == 0:
            args.outputFname = None
        else:
            args.outputFname = os.path.splitext(args.inputFname)[0] + ".csv"
    print(f"Processing video {args.inputFname}")
    print(f"Outputs will be saved to {args.outputFname}")

from collections import deque
from multiprocessing.pool import ThreadPool
import tensorflow as tf
from tensorflow.python.keras import optimizers, models
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
import cv2 as cv
from feat.utils import get_resource_path


def face_rect_to_coords(rectangle):
    """
    Takes in a (x, y, w, h) array and transforms it into (x, y, x2, y2)
    """
    return [
        rectangle[0],
        rectangle[1],
        rectangle[0] + rectangle[2],
        rectangle[1] + rectangle[3],
    ]


# load pre trained emotion model
print("Loading FEX DCNN model.")
dcnn_model_path = os.path.join(get_resource_path(), "fer_aug_model.h5")
if not os.path.exists(dcnn_model_path):
    print("Emotion prediction model not found. Please run download_models.py.")
model = models.load_model(dcnn_model_path)  # Load model to use.
(_, img_w, img_h, img_c) = model.layers[0].input_shape  # model input shape.
mapper = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}
emotion_columns = [key for key in mapper.values()]

print("Loading Face Detection model.")
face_detection_model_path = cv.data.haarcascades + "haarcascade_frontalface_alt.xml"
if not os.path.exists(face_detection_model_path):
    print(
        "Face detection model not found. Check haarcascade_frontalface_alt.xml exists in your opencv installation (cv.data)."
    )
face_cascade = cv.CascadeClassifier(face_detection_model_path)
facebox_columns = ["facebox_x", "facebox_y", "facebox_w", "facebox_h"]
facebox_empty = np.empty((1, 4))
facebox_empty[:] = np.nan
empty_facebox = pd.DataFrame(facebox_empty, columns=facebox_columns)

print("Loading Face Landmark model.")
face_landmark = cv.face.createFacemarkLBF()
landmark_model_path = os.path.join(get_resource_path(), "lbfmodel.yaml")
if not os.path.exists(landmark_model_path):
    print("Face landmark model not found. Please run download_models.py.")
face_landmark.loadModel(landmark_model_path)
landmark_columns = (
    np.array([(f"x_{i}", f"y_{i}") for i in range(68)]).reshape(1, 136)[0].tolist()
)
landmark_empty = np.empty((1, 136))
landmark_empty[:] = np.nan
empty_landmark = pd.DataFrame(landmark_empty, columns=landmark_columns)

# create empty df for predictions
predictions = np.empty((1, len(mapper)))
predictions[:] = np.nan
empty_df = pd.DataFrame(predictions, columns=mapper.values())

frame_columns = ["frame"]
init_df = pd.DataFrame(
    columns=frame_columns + emotion_columns + facebox_columns + landmark_columns
)
if args.outputFname:
    init_df.to_csv(args.outputFname, index=False, header=True)


def process_frame(frame, counter, input_image_size=(48, 48), mapper=mapper):
    """
    Takes a frame from OpenCV and prepares it as a tensor to be predicted by model.

    frame: Image input.
    counter: Frame number.
    """
    try:
        # change image to grayscale
        grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # find faces
        detected_faces = face_cascade.detectMultiScale(grayscale_image)
        # detect landmarks
        ok, landmarks = face_landmark.fit(grayscale_image, detected_faces)
        landmarks_df = pd.DataFrame(
            landmarks[0][0].reshape(1, 136), columns=landmark_columns, index=[counter]
        )
        # Use Tiankang's colde to align the faces to the center

        # crop just the face area
        if len(detected_faces) > 0:
            facebox_df = pd.DataFrame(
                [detected_faces[0]], columns=facebox_columns, index=[counter]
            )

            grayscale_cropped_face = Image.fromarray(grayscale_image).crop(
                face_rect_to_coords(detected_faces[0])
            )
            # resize face to newsize 48 x 48
            # print("resizeface", grayscale_cropped_face.shape, img_w, img_h, img_c)
            grayscale_cropped_resized_face = grayscale_cropped_face.resize(
                input_image_size
            )
            # reshape to put in model
            grayscale_cropped_resized_reshaped_face = np.array(
                grayscale_cropped_resized_face
            ).reshape(1, img_w, img_h, img_c)
            # normalize
            normalize_grayscale_cropped_resized_reshaped_face = (
                grayscale_cropped_resized_reshaped_face / 255.0
            )
            # make tensor
            tensor_img = tf.convert_to_tensor(
                normalize_grayscale_cropped_resized_reshaped_face
            )
            # make predictions
            predictions = model.predict(tensor_img)
            emotion_df = pd.DataFrame(
                predictions, columns=mapper.values(), index=[counter]
            )
            return (
                emotion_df,
                frame,
                facebox_df.reindex(index=[counter]),
                landmarks_df.reindex(index=[counter]),
            )
    except:
        return (
            empty_df.reindex(index=[counter]),
            frame,
            empty_facebox.reindex(index=[counter]),
            empty_landmark.reindex(index=[counter]),
        )


if __name__ == "__main__":
    # Setup.
    cap = cv.VideoCapture(args.inputFname)
    fps = cap.get(cv.CAP_PROP_FPS)  # get frames per second.

    # Determine whether to use multiprocessing.
    if args.n_jobs == -1:
        thread_num = cv.getNumberOfCPUs()  # get available cpus
    else:
        thread_num = args.n_jobs
    pool = ThreadPool(processes=thread_num)

    pending_task = deque()

    counter = 0
    processed_frames = 0
    detected_faces = []
    print("Processing video.")
    while True:
        # Consume the queue.
        while len(pending_task) > 0 and pending_task[0].ready():
            emotion_df, frame, facebox_df, landmarks_df = pending_task.popleft().get()
            df = pd.concat([emotion_df, facebox_df, landmarks_df], axis=1)
            if args.verbose:
                print(emotion_df)
            # Save to output file.
            if args.outputFname:
                df.to_csv(args.outputFname, index=True, header=False, mode="a")
            processed_frames = processed_frames + 1
            # if we are in demo mode.
            if args.inputFname == 0:
                # if a face exists
                detected_faces = facebox_df.values
                if not np.isnan(detected_faces[0]).any():
                    # draw rectangle on face.
                    (x, y, xw, yh) = face_rect_to_coords(detected_faces[0])
                    cv.rectangle(frame, (x, y), (xw, yh), (0, 255, 0))
                    labels = [
                        key + ": " + str(np.round(df.loc[:, key].values[0], 2))
                        for key in mapper.values()
                    ]
                    y0, dy = 50, 30
                    for i, line in enumerate(labels):
                        y = y0 + i * dy
                        cv.putText(
                            frame,
                            line,
                            (30, y),
                            cv.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            3,
                        )
                else:
                    cv.putText(
                        frame,
                        "No Face Found",
                        (30, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3,
                    )
                # Show preview.
                cv.imshow("Emotion detector demo", frame)
                if cv.waitKey(1) == 27:
                    break

        # Populate the queue.
        if len(pending_task) < thread_num:
            frame_got, frame = cap.read()
            # Process at every seconds.
            if counter % args.skip_frames == 0:
                if frame_got:
                    task = pool.apply_async(process_frame, (frame.copy(), counter))
                    pending_task.append(task)
            counter = counter + 1

        if not frame_got:
            break
cap.release()
cv.destroyAllWindows()
