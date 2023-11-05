# Mouse Neuronal Response to Dynamic Stimuli Prediction Model

## Description

The Mouse Neuronal Response to Dynamic Stimuli Prediction Model is a project that leverages a fine-tuned Video Vision Transformer (ViVIT) to predict murine neuronal activity based on video frames displayed to the mouse.

## Installation

To install and set up this project, you'll need to perform the following steps:

1. Clone the repository to your local machine:
git clone https://github.com/your-username/mouse-neuronal-prediction.git


2. Install the required dependencies, including ViVIT and qLora fine-tuning modules.

## Usage

To use this prediction model, follow these steps:

1. Preprocess the video frames and organize them in the required format.
2. Load the pre-trained model and pass the frames through it.
3. Obtain predictions for murine neuronal activity.

Example code snippet:

```python
# Load and preprocess video frames
frames = preprocess_frames('path_to_video_frames/')

# Load the pre-trained model
model = load_model('path_to_pretrained_model/')

# Predict neuronal activity
predictions = model.predict(frames)

# Process predictions as needed
```
### Contributing

We welcome contributions! If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository and create your branch from `main`.
2. Make sure to adhere to the project's coding style and guidelines.
3. Submit a pull request with a detailed description of your changes.

### License

This project is licensed under the [MIT License](LICENSE).

### Acknowledgements

Special thanks to [Sensorium 2023](https://www.sensorium-competition.net/) for their support and contributions to this project.

