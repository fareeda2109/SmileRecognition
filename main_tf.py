# %% imports
import os
import numpy as np
import tensorflow as tf

from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse
from collections import Counter
import time


# %%
def create_dataset_new(data_path, batch_size=16, plot=False, verbose=False):
    # fetch images from data_path/files
    # fetch labels from data_path/labels.txt

    images_path = os.path.join(data_path, "files")
    labels_path = os.path.join(data_path, "labels.txt")
    images = []
    labels = []
    print(f"Reading labels from {labels_path}...")
    with open(labels_path) as f:
        labels = f.readlines()

    print(f"labels: {len(labels)}\n Sample: {labels[0]}")

    # discard other annotations
    labels = [int(label.split(" ")[0]) for label in labels]

    # 4000 :: 1 :: 0
    print(f"(processed) labels: {len(labels)} :: {labels[0]} :: {labels[-1]}")

    print(f"Reading images from {images_path}...")
    # scan images_path for image files using glob (.jpg, .png)
    images_files = glob(images_path + "/*.jpg")
    # filenames: file0001.jpg, file0002.jpg, ... file4000.jpg
    images_files = sorted(images_files, key=lambda x: int(x.split("/")[-1][4:-4]))

    print(f"Found images: {len(images_files)}\n Sample: {images_files[:20]}")

    # read images
    images = []
    for image_file in images_files:
        image = Image.open(image_file)
        # resize to 64x64
        image = image.resize((64, 64)).convert("RGB")
        image = np.array(image)
        images.append(image)

    images = np.array(images)
    images = images / 255.0

    # upto 2162 smiling, 1838 not smiling
    labels = np.array(labels)

    # plot before shuffle
    images, labels, images_files = shuffle(images, labels, images_files, random_state=0)

    # plot random samples with labels
    if plot:
        plt.figure(figsize=(10, 10))

        n = 10

        for i in range(n):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i])
            plt.title(
                f"Smiling: {labels[i]} {images_files[i].split('file')[-1].replace('.jpg', '')}"
            )
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("random_samples_before_shuffle.png")
        plt.close()

    print(f"images: {images.shape}, labels: {labels.shape}")

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=0
    )

    print(
        f"train_images: {train_images.shape}, train_labels: {train_labels.shape}, test_images: {test_images.shape}, test_labels: {test_labels.shape}"
    )

    # convert to tf.data.Dataset

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_ds = train_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    print(f"Number of batches in train_ds: {len(train_ds)}")
    print(f"Number of batches in test_ds: {len(test_ds)}")

    # print sample
    for image_batch, label_batch in train_ds.take(1):
        print(f"Image batch shape: {image_batch.shape}")
        print(f"Label batch shape: {label_batch.shape}")
        if plot:
            # plot images in batch
            plt.figure(figsize=(10, 10))

            for i in range(batch_size):
                # calculate number of rows needed plot 8 columns
                rows = batch_size // 4 + 1
                plt.subplot(rows, 4, i + 1)
                plt.imshow(image_batch[i])
                plt.title(f"Smiling: {label_batch[i]}")
                plt.axis("off")
            plt.tight_layout()
            plt.savefig("random_samples_after_shuffle_one_batch.png")
            plt.close()

    return train_ds, test_ds


# %%
def preprocess_image(image_path, verbose=False):
    if type(image_path) == str:
        # img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
        # img = tf.keras.preprocessing.image.img_to_array(img)
        img = Image.open(image_path)
        # resize to 64x64
        img = img.resize((64, 64)).convert("RGB")
        img = np.array(img).astype(np.float32)
        img = img / 255.0
    else:
        # cv2 image
        img = image_path
        img_tensor = tf.convert_to_tensor(img)
        img_tensor = tf.image.resize(img_tensor, [64, 64])
        img = tf.expand_dims(img_tensor, axis=0)

        img /= 255.0

    if verbose:
        print(f"image shape: {img.shape}")

    return img


# %%
def plot_class_distribution(dataset, class_names, dtype="train", verbose=False):
    print(f'Plotting "{dtype}" class distribution...')
    # count number of classes in train and test using filter
    classes = []
    for _, y in dataset:
        classes.extend(y.numpy().tolist())

    print(f"counter: {Counter(classes)}")
    # plot histogram of Counter(whole dataset) using subplot
    plt.figure(figsize=(10, 6))
    # plot all, train, and test in one figure using subplot
    plt.bar(Counter(classes).keys(), Counter(classes).values())
    plt.title(f"{dtype.title()} classes distribution")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.xticks(list(Counter(classes).keys()))

    plt.yticks(
        np.arange(len(classes), step=min(500, max(Counter(classes).values()) // 4))
    )

    plt.tight_layout()
    plt.savefig(f"{dtype}_classes_distribution.png")
    # plt.show()
    plt.close()


# %%
def infer(model, img, verbose=False):
    # predict
    start = time.time()
    pred = model.predict(img, verbose=0)
    pred = pred.squeeze()
    end = time.time()
    took_s = end - start
    took_ms = took_s * 1000
    took = f"{took_s:.2f}s or {took_ms:.2f}ms"
    if verbose:
        print(f"pred: {pred} :: shape: {pred.shape}")

    return pred, took


# %%
def plot_confusion_matrix(model, test_dataset, class_names, verbose=False):
    y_true = []
    y_pred = []

    for batch in test_dataset:
        images, labels = batch
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        preds = preds.round().astype(int)
        y_pred.extend(preds.squeeze().tolist())

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax)

    plt.title("Confusion Matrix")
    # save confusion matrix
    plt.savefig("confusion_matrix.png")
    plt.show()
    # Save the figure
    fig.savefig("confusion_matrix.png")


# %%
def save_model(model, model_path="smile_model", verbose=False):
    # save model in saved_model format

    model.save(f"{model_path.h5}")

    if verbose:
        print(f"model saved at {model_path}")


def load_model(model_path="smile_model", verbose=False):
    # load model
    model = tf.keras.models.load_model(model_path)

    # warm up model
    for _ in range(3):
        _ = model.predict(tf.random.normal((1, 64, 64, 3)), verbose=0)

    if verbose:
        print(f"model loaded from {model_path}")

    return model


def decode_prediction(pred, verbose=False):
    # decode prediction
    pred = pred.round().astype(int)

    class_names = ["Not Smiling", "Smiling"]
    pred_class = class_names[pred]

    if verbose:
        print(f"pred_class: {pred_class}")

    return pred_class


# %%
def infer_from_image(model, image_path, verbose=False):
    # preprocess image
    img = preprocess_image(image_path, verbose=verbose)
    # infer
    img = tf.expand_dims(img, axis=0)
    pred, took = infer(model, img, verbose=verbose)
    # decode prediction
    pred_class = decode_prediction(pred, verbose=verbose)
    print(f"Took {took}")
    if verbose:
        plt.imshow(img[0])
        plt.title(f"Prediction: {pred_class} in {took}")
        plt.show()

    return pred, pred_class


def haar_cascade_face_detect(frame):
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # load haar cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print("No faces detected...")
        return frame
    # sort faces by area descending and pick the first one
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    big_face = faces[0]

    # expand face by 20%
    expand_by = 0.2
    x, y, w, h = big_face
    x = max(int(x - expand_by * w), 0)
    y = max(int(y - expand_by * h), 0)
    w = int(w + (2 * expand_by * w))
    h = int(h + (2 * expand_by * h))

    # Ensure the expanded rectangle is within frame bounds
    x_end = min(x + w, frame.shape[1])
    y_end = min(y + h, frame.shape[0])

    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)

    # Crop frame
    face_frame = frame[y:y_end, x:x_end]

    # Assert face_frame is a color image and shape does not contain 0
    assert (
        len(face_frame.shape) == 3 and 0 not in face_frame.shape
    ), "Invalid face frame"

    return face_frame


# %%
def infer_from_video(model, video_path, with_face_detection, verbose=False):
    print(f'Running inference on video "{video_path}"...')
    print(f'Press "q" to quit...')
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1000 / fps

    inference_time = 50  # milliseconds

    # number of frames to skip
    n = int(inference_time // frame_time)
    frame_count = 0
    # wait key = q
    wait_key = ord("q")
    i = 0
    while True:
        print(f"{i} frames processed...")
        i = i + 1
        # read frame
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame...")
            break
        if with_face_detection:
            face_frame = haar_cascade_face_detect(frame)
        else:
            face_frame = frame
        # BGR to RGB
        x = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        # frame = cv2.resize(frame, (64, 64))
        if frame_count % (n + 1) == 0:
            x = preprocess_image(x)
            # get prediction
            pred, took = infer(model, x, verbose=verbose)
        pred_text = (
            f"Prediction ({pred :.2f}): {decode_prediction(pred)} in {took}"
            if pred is not None
            else ""
        )

        # show image
        cv2.putText(
            frame,
            f"{pred_text} -- Press q to quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("frame", frame)
        frame_count = frame_count + 1

        # check for stop key
        if cv2.waitKey(1) & 0xFF == wait_key:
            break
    cap.release()
    cv2.destroyAllWindows()


# %%
def infer_from_webcam(model, webcam=0, with_face_detection=False, verbose=False):
    print(f'Running inference on webcam "{webcam}"...')
    print(f'Press "q" to quit...')
    cap = cv2.VideoCapture(webcam)

    # wait key = q
    wait_key = ord("q")
    while True:
        # read frame
        ret, frame = cap.read()

        # downsample frame
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        if not ret:
            print("Error reading frame...")
            break

        # flip frame
        frame = cv2.flip(frame, 1)
        # detect face
        if with_face_detection:
            face_frame = haar_cascade_face_detect(frame)
        else:
            face_frame = frame

        # BGR to RGB
        x = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        # x = cv2.resize(frame, (64, 64))

        x = preprocess_image(x)
        # get prediction
        pred, took = infer(model, x, verbose=verbose)
        # show image
        cv2.putText(
            frame,
            f"Prediction ({pred :.2f}): {decode_prediction(pred)} in {took}\nPress q to quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("frame", frame)

        # check for stop key
        if cv2.waitKey(1) & 0xFF == wait_key:
            break
    # release camera
    cap.release()
    # close all windows
    cv2.destroyAllWindows()


# %%
def build_model(verbose=False):
    # create model
    l2_reg = 1e-4
    # build model
    model = tf.keras.Sequential(
        [
            # # data augmentation used in google colab, didn't work on local machine
            # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            # tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
            tf.keras.layers.Input(shape=(64, 64, 3)),
            # input: (3, 64, 64) -> (32, 64, 64)
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(1),
            # input: (32, 64, 64) -> (32, 32, 32)
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(2),
            # input: (32, 32, 32) -> (32, 32, 32)
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(1),
            # input: (32, 32, 32) -> (32, 16, 16)
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(2),
            # input: (32, 16, 16) -> (32, 8, 8)
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            ),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    # warm up model
    for _ in range(3):
        pred = model.predict(tf.random.normal((1, 64, 64, 3)), verbose=0)

    if verbose:
        print(f"model summary: {model.summary()}")
        print(pred.shape)

    return model


def augment_dataset(dataset, verbose=False):
    dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    # dataset = dataset.map(lambda x, y: (tf.image.random_flip_up_down(x), y))
    dataset = dataset.map(lambda x, y: (tf.image.rot90(x), y))
    dataset = dataset.map(
        lambda x, y: (tf.image.random_brightness(x, max_delta=0.1), y)
    )
    dataset = dataset.map(
        lambda x, y: (tf.image.random_contrast(x, lower=0.1, upper=0.2), y)
    )
    dataset = dataset.map(lambda x, y: (tf.image.random_hue(x, max_delta=0.1), y))
    dataset = dataset.map(
        lambda x, y: (tf.image.random_saturation(x, lower=0.1, upper=0.2), y)
    )

    # dataset = dataset.map(lambda x, y: (tf.image.random_crop(x, size=[64, 64, 3]), y))
    # dataset = dataset.map(
    #     lambda x, y: (tf.image.central_crop(x, central_fraction=0.5), y)
    # )
    # make sure all images are of size 64x64
    # dataset = dataset.map(lambda x, y: (tf.image.resize(x, [64, 64]), y))

    return dataset


def plot_one_batch_from_test(
    model, test_dataset, class_names=["Not Smiling", "Smiling"]
):
    plt.figure(figsize=(10, 10))
    for images, labels in test_dataset.skip(3).take(1):
        preds, _ = infer(model, images, verbose=False)
        predicted_labels = [decode_prediction(pred) for pred in preds]
        gt_labels = labels.numpy()
        # evaluate on one batch
        eval_res = model.evaluate(images, labels, verbose=0)
        acc_per_batch = f"Acc per batch: {eval_res[1] * 100 :.2f} %"
        n = len(images)

        for i in range(n):
            rows = n // 4 + 1
            plt.subplot(rows, 4, i + 1)
            plt.imshow(images[i])
            plt.title(f"Pred: {predicted_labels[i]}, GT: {class_names[gt_labels[i]]}")
            plt.axis("off")
        plt.suptitle(acc_per_batch)
        plt.tight_layout()

    plt.savefig("model_performance_on_test.png")
    plt.close()


# %%
def parse_args():
    print(f"Parsing arguments...")
    parser = argparse.ArgumentParser(description="Smile Detector")

    parser.add_argument("-t", "--train", action="store_true", help="Train model")
    parser.add_argument(
        "-eval", "--evaluate", action="store_true", help="Evaluate model"
    )
    parser.add_argument(
        "-e", "--epoches", type=int, default=50, help="Epoches (50 is default)"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )

    # required for the above to work
    parser.add_argument("-d", "--data_path", type=str, help="Data path")

    parser.add_argument("-i", "--infer", action="store_true", help="Infer on image")

    parser.add_argument(
        "-w", "--webcam", type=int, default=0, help="Webcam ID (default: 0)"
    )
    parser.add_argument("-f", "--img_path", type=str, help="Infer on image file")
    parser.add_argument("-v", "--video_path", type=str, help="Infer on video file")
    # with face detection
    parser.add_argument(
        "-fd",
        "--face_detection",
        action="store_true",
        default=False,
        help="Use face detection to infer",
    )
    # required for inference
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="smile_model",
        help="Model path for [inference only]",
    )

    parser.add_argument(
        "-exp",
        "--explore_training",
        action="store_true",
        help="Explore training with plots",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    args = parser.parse_args()
    print(f"args: {args}")
    return args


# %%
if __name__ == "__main__":
    args = parse_args()
    # set seeds
    np.random.seed(1)
    tf.random.set_seed(1)
    with_face_detection = args.face_detection
    if args.train or args.evaluate:
        if args.data_path is None:
            print(f"Please provide data_path")
            exit(1)
        BATCH_SIZE = 32

        # create dataset
        train_dataset, test_dataset = create_dataset_new(
            args.data_path,
            batch_size=BATCH_SIZE,
            plot=args.explore_training,
            verbose=args.verbose,
        )
        class_names = ["Not Smiling", "Smiling"]
        if args.train:
            train_dataset = augment_dataset(train_dataset, verbose=args.verbose)

            # if args.explore_training:
            # plot class distribution
            plot_class_distribution(
                train_dataset, class_names, dtype="train", verbose=args.verbose
            )
            plot_class_distribution(
                test_dataset, class_names, dtype="test", verbose=args.verbose
            )

            # build model
            model = build_model(verbose=args.verbose)

            initial_learning_rate = args.learning_rate
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=100000,
                decay_rate=0.96,
                staircase=True,
            )

            # compile model
            model.compile(
                optimizer=tf.keras.optimizers.Nadam(learning_rate=lr_schedule),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # train model
            EPOCHS = args.epoches
            history = model.fit(
                train_dataset, epochs=EPOCHS, validation_data=test_dataset
            )

            # if args.explore_training:
            # plot learning curve using pandas
            df = pd.DataFrame(history.history)
            ax = df.plot(y=["loss", "accuracy", "val_loss", "val_accuracy"])

            # Use plt.gcf() to get the current figure if it was not assigned to an object
            fig = plt.gcf()
            # Save the current figure
            plt.title(
                f"Learning Curve with lr={args.learning_rate} and epochs={args.epoches}"
            )
            plt.xlabel("Epochs")
            plt.ylabel("Loss/Accuracy")
            fig.savefig(
                "learning_curve.png"
            )  # You can specify a path here if necessary
            plot_confusion_matrix(
                model, test_dataset, class_names, verbose=args.verbose
            )
            model_path = args.model_path
            # save model
            save_model(model, model_path=model_path, verbose=args.verbose)
            print(f"Training completed.")
            # evaluate model
            loss, acc = model.evaluate(test_dataset)
            print(f"Test loss: {loss}, Test accuracy: {acc * 100 :.2f}")
            print(f"Launching webcam to test...")
            # infer from webcam
            infer_from_webcam(model, webcam=args.webcam, verbose=args.verbose)
        else:
            # load model
            model = load_model(model_path=args.model_path, verbose=args.verbose)
            # evaluate model
            loss, acc = model.evaluate(test_dataset)
            print(f"Test loss: {loss}, Test accuracy: {acc * 100 :.2f}")
            plot_one_batch_from_test(model, test_dataset, class_names)

    elif args.infer:
        if args.model_path is None:
            print(f"Please provide model_path")
            exit(1)

        if args.webcam is None and args.img_path is None and args.video_path is None:
            if args.img_path is None:
                print("Please specify image path to infer on...")
                exit(0)
            elif args.video_path is None:
                print("Please specify video path to infer on...")
                exit(0)
            else:
                print("Using default webcam id: 0")

        else:
            # load model
            model = load_model(model_path=args.model_path, verbose=args.verbose)

            if args.img_path is not None:
                # infer from image
                infer_from_image(model, image_path=args.img_path, verbose=args.verbose)
            elif args.video_path is not None:
                # infer from video
                infer_from_video(
                    model, args.video_path, with_face_detection, verbose=args.verbose
                )
            else:
                # infer from webcam
                infer_from_webcam(
                    model,
                    webcam=args.webcam,
                    with_face_detection=with_face_detection,
                    verbose=args.verbose,
                )

    else:
        print("Please specify either --train or --infer")
        exit(0)
