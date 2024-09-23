# 🤟 ASL-Conversion

This project builds a Long Short-Term Memory (LSTM) neural network that translates American Sign Language (ASL) gestures to text in real-time using [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic), a comprehensive human body detection solution by Google.

## 🧠 Abstract

Machine learning (ML) is a branch of artificial intelligence that focuses on creating algorithms and models enabling computers to learn and make predictions or decisions without explicit programming. Neural networks, designed to mimic the structure and function of the human brain, consist of interconnected nodes (artificial neurons) organized into layers. Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN), differ from traditional RNNs by utilizing memory cells and gates to selectively retain or forget information over time. This makes LSTMs especially useful for tasks involving sequential or time-series data.

This project explores the application of machine learning and computer vision to gesture recognition. By using deep learning models alongside Google's MediaPipe framework (a powerful computer vision framework for human pose, hand tracking, and facial recognition), we detect and classify human gestures in real time, focusing specifically on American Sign Language.

## 🛠️ Features

- **Real-time ASL to Text Translation:** Converts ASL gestures into text using an LSTM neural network.
- **MediaPipe Integration:** Utilizes MediaPipe Holistic to perform human pose and hand gesture detection.
- **Deep Learning Model:** LSTM neural network trained to recognize sequential patterns in ASL gestures.
- **Gesture Recognition:** Recognizes human gestures by tracking hand and body key points.
- **Real-time Classification:** Displays classification results instantly using the trained LSTM model.

## 🚀 Deployment

This project uses Jupyter Notebooks for model training and real-time gesture classification. Follow the steps below to set it up locally.

### 1. Clone the Repository

```bash
git clone https://github.com/GuptaDiva/asl-conversion.git
cd asl-conversion
```

### 2. Install Dependencies

Ensure all required dependencies are installed using the provided `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

The dataset of gesture sequences is stored in the `Dataset` folder. Ensure that the dataset is properly formatted and ready for training.

```
📂Dataset
 ┣ 📂train               # Training data
 ┗ 📂test                # Testing data
```

### 4. Train the LSTM Model

Use the Jupyter Notebook `Neural Network.ipynb` to train the LSTM model:

```bash
jupyter notebook Neural Network.ipynb
```

### 5. Real-time Gesture Classification

Once the model is trained, use the notebook `Neural Network90.h5` or `Neural Network.h5` to perform real-time gesture classification with MediaPipe Holistic.

## 📂 Project Structure

```
📦asl-conversion
 ┣ 📂.ipynb_checkpoints     # Jupyter Notebook checkpoints
 ┣ 📂Dataset                # Training and testing datasets
 ┣ 📂Logs/train             # Logs from model training
 ┣ 📜0.npy                  # Example numpy array file for storing data
 ┣ 📜LICENSE                # License for the project
 ┣ 📜Neural Network.h5      # Trained LSTM model
 ┣ 📜Neural Network90.h5    # Another checkpoint of the trained model
 ┣ 📜Neural Network.ipynb   # Jupyter Notebook for training the model
 ┣ 📜README.md              # Project documentation
 ┗ 📜requirements.txt       # Python dependencies
```

## 📝 How It Works

### Data Processing

The dataset contains ASL gesture data, processed and stored as NumPy arrays (`0.npy`) that contain key points corresponding to different hand gestures. MediaPipe Holistic extracts key points for hands, body, and face, which are used as input for the neural network.

### LSTM Neural Network

- **Architecture:** The LSTM model is designed to learn sequential patterns from gesture data.
- **Training:** The model is trained using the data from the `Dataset` folder, which contains labeled sequences of ASL gestures.
- **Real-Time Inference:** After training, the model predicts and classifies hand gestures in real time based on MediaPipe's key point detection.

### MediaPipe Integration

- **Key Point Detection:** MediaPipe Holistic is used to detect key points for the hand, face, and body in real-time video input.
- **Gesture Recognition:** The detected key points are fed into the LSTM model to classify ASL gestures.

## 📦 Dependencies

This project requires the following dependencies, which are listed in the `requirements.txt`:

```
tensorflow
numpy
mediapipe
matplotlib
opencv-python
```

## 🧑‍💻 Author

- **Diva Gupta** – [LinkedIn](https://www.linkedin.com/in/diva-gupta-a6107421b/)
                 – [Github](https://github.com/GuptaDiva)