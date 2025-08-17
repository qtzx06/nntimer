# datasets

the training data used for this project is not included in the git repository to keep it lightweight.

## structure

the project expects the following directory structure for training:

- `datasets/cube_data/state/solved/`: contains images of solved cubes.
- `datasets/cube_data/state/unsolved/`: contains images of unsolved cubes.
- `datasets/cube_data/vision/`: contains the yolo format dataset for object detection.

## to train the models:

1.  populate the directories above with your own images.
2.  run `python train_classifier.py` to train the solved/unsolved cnn.
3.  run `python train_detector.py` to train the yolo object detector.
