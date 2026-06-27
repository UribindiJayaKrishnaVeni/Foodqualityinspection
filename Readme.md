# FreshScan — AI-Powered Food Quality Inspection System

FreshScan is a deep learning–based web application that identifies the type of a food item (fruit, vegetable, bread, dairy, or meat) and predicts whether it is **fresh** or **spoiled** directly from an uploaded image.

## Overview

Manual inspection of food freshness is subjective, inconsistent, and time-consuming. FreshScan automates this process using a Convolutional Neural Network trained with transfer learning, wrapped in a simple web interface where a user uploads a photo and instantly receives a freshness prediction with a confidence score.

## Features

- Classifies images into **36 categories** (18 food types × Fresh/Rotten)
- Covers fruits, vegetables, bread, dairy, and meat
- Returns a confidence score for every prediction
- Flags images that don't match any trained category as **"Unknown"** instead of forcing an incorrect label
- Clean, responsive web interface with drag-and-drop image upload

## Tech Stack

| Component | Technology |
|---|---|
| Core Language | Python 3.11.9 |
| Deep Learning Framework | TensorFlow 2.21.0 |
| Model API | Keras |
| Pretrained Backbone | MobileNetV2 |
| Backend | Flask |
| Image Processing | Pillow (PIL) |
| Numerical Processing | NumPy |
| Frontend | HTML5, CSS3, JavaScript |
| Development Environment | Jupyter Notebook, Visual Studio Code |
| Dataset Source | [Kaggle — Food Fresh Detection](https://www.kaggle.com/datasets/zhenqi123/food-fresh-detection) |

## Dataset

- **23,910 images** across 36 classes
- Split: 70% train / 15% validation / 15% test
- Categories: Apple, Banana, Bell Pepper, Bitter Gourd, Bread, Broccoli, Capsicum, Carrot, Cucumber, Dairy, Mango, Meat, Okra, Orange, Potato, Spinach, Strawberry, Tomato — each with Fresh and Rotten classes

## Model Architecture

```
MobileNetV2 (pretrained on ImageNet, frozen base)
        ↓
GlobalAveragePooling2D
        ↓
Dense(256, activation='relu')
        ↓
Dropout(0.4)
        ↓
Dense(36, activation='softmax')
```

Transfer learning was used to leverage pretrained visual features, reducing training time while maintaining high accuracy.

## Results

- Training accuracy: ~97%
- Validation accuracy: ~95%
- Training and validation curves converge closely, indicating a good fit (no significant overfitting or underfitting)

## Project Structure

```
food-quality-inspection/
│
├── app.py                              # Flask backend
├── requirements.txt                    # Python dependencies
├── class_names.json                    # 36 class labels
├── food_quality_model_36class.keras    # Trained model
│
├── templates/
│   └── index.html                      # Frontend (HTML + CSS + JS)
│
└── static/
    └── uploads/                        # Temporary storage for uploaded images
```

> **Note:** The training dataset (`data/`, `data_split/`) and the original Kaggle archive are not included in this repository due to size constraints (~2GB+). See [Dataset](#dataset) above for the source.

## Setup & Installation

1. Clone the repository
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application
   ```bash
   python app.py
   ```

4. Open your browser and go to
   ```
   http://127.0.0.1:5000
   ```

## How It Works

1. User uploads a food image through the web interface
2. The image is sent to the Flask backend via a `/predict` API route
3. The image is resized to 224×224 and normalized
4. The trained model predicts class probabilities across 36 categories
5. If the top confidence is **≥ 90%**, the predicted food type and freshness status are shown with the confidence score
6. If confidence is **below 90%**, the result is shown as **"Unknown Image"** instead of a forced incorrect label

## Limitations

- Some classes (e.g. Broccoli, Spinach) have fewer training images than others, which may reduce accuracy for those specific categories
- Prediction quality depends on image clarity, lighting, and angle
- The 90% confidence threshold for "Unknown" detection is a heuristic, not a guaranteed out-of-distribution detector
- Currently limited to the 36 trained categories

## Future Improvements

- Expand dataset for underrepresented classes
- Add more food categories
- Deploy backend (Render) and frontend (Vercel) for public access
- Add image history / inspection logs

## References

- Dataset: [Kaggle — Food Fresh Detection](https://www.kaggle.com/datasets/zhenqi123/food-fresh-detection)
- [TensorFlow Documentation](https://www.tensorflow.org)
- [Keras Applications — MobileNetV2](https://keras.io/api/applications/mobilenet/)
- [Flask Documentation](https://flask.palletsprojects.com)
- Sandler, M., et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks* (2018)

## License

This project was developed for academic purposes.