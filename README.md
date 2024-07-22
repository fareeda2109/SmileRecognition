Smile detector using deep learning with real-time prediction

To train the model, use -t (--train) to set it to train mode, -d (--data) to set the data directory, -e (--epochs) to set the number of epochs, -lr (--learning_rate) to set the learning rate, -m (--model_path) to set the model name, and --verbose to set the verbose mode.

if packages are not installed, run:

```bash
pip3 install -r requirements.txt
```

```bashs
python3 main_tf.py -t -d 'data' -e 10 -lr 0.0001 -m 'smile_rec_model.h5' --verbose
```

add -exp (--explore_training) to set the explore training mode

### model is trained on google colab, then model is downloaded. It was trained for 180 epoches, with exponential decaying learing rate of 1e-3

```bash
python3 main_tf.py -t -d 'data' -e 10 -lr 0.0001 -m 'smile_rec_model.h5' -exp --verbose
```

To evaluate the model with 20% dataset

```bash
python3 main_tf.py -eval -d 'data' -m 'smile_rec_model.h5' --verbose
```

---

To run inference, use -i (--infer) to set it to inference mode, -m (--model_path) to set the model name, and --verbose to set the verbose mode.

```bash
python3 main_tf.py -i -m 'smile_rec_model.h5' --verbose
```

Then choose source;
From image:

```bash
python3 main_tf.py -i -m 'smile_rec_model.h5' -f 'img.jpg' --verbose
```

From video:

```bash
python3 main_tf.py -i -m 'smile_rec_model.h5' -v 'video.mp4' --verbose
```

From webcam: (0 - pc webcam, >=1 - external webcam)

```bash
python3 main_tf.py -i -m 'smile_rec_model.h5' -w 0 --verbose
```
