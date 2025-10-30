# 🧠 CIFAR-10 Image Classification App

This project demonstrates a **complete machine learning pipeline** — from training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset to deploying it using **Streamlit** for real-time image classification.

---

📂 Dataset
To run the project, the CIFAR-10 dataset will be downloaded automatically when you run `preprocess.py`.  
You don’t need to upload any dataset files to the repository.


## 🚀 Project Structure

```
src/
│
├── preprocess.py      # Load and preprocess the CIFAR-10 dataset
├── model.py              # Build, train, and save the CNN model
├── visualizations.py      # Plot training and test results
├── app.py                # Streamlit app for live image prediction
│
├── data/                 # Stores any exported or processed data
├── models/               # Stores the trained model (.h5 file)
│
├── requirements.txt      # Required dependencies
└── README.md             # Project documentation
```

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mazenmuhamed14/cifar10-classifier.git
   cd cifar10-classifier
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # On Windows
   source venv/bin/activate   # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Training the Model

Run the following script to train and save the CNN model:

```bash
python src/model.py
```

Once training finishes, the model will be saved automatically in:
```
models/cifar10_cnn_model.h5
```

---

## 🌐 Running the Streamlit App

Launch the web app using:
```bash
streamlit run src/app.py
```

Then open the provided local URL in your browser.

Upload an image (e.g., a car, cat, or airplane) and the model will:
- Preprocess the image
- Predict its class
- Display confidence percentage

---

## 📊 CIFAR-10 Dataset Classes

The dataset includes 10 categories:

```
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

---

## 🧩 Example Output

When you upload a picture of a dog, the app might display:

```
Predicted Class: Dog
Confidence: 93.76%
```

---

## 🧱 Future Improvements

- Add Data Augmentation for better accuracy.
- Use Transfer Learning (ResNet, VGG, etc.)
- Add image explanation (Grad-CAM visualization).
- Deploy on Hugging Face or Streamlit Cloud.

---

## 🧑‍💻 Author
**Mazen Mohamed**  
📚 Data Science & Front-End Developer  
🚀 Project built using TensorFlow, Keras, NumPy, and Streamlit.
