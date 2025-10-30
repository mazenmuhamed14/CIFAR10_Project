import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 1️⃣ Load the trained model
MODEL_PATH = "models/cifar10_cnn_model.h5"
model = load_model(MODEL_PATH)

# CIFAR-10 class names
classes = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# 2️⃣ Streamlit Page Setup
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🧠", layout="centered")
st.title("🧠 CIFAR-10 Image Classifier")
st.markdown("Upload an image of one of the following classes:")
st.write(", ".join(classes))

# 3️⃣ File Upload Section
uploaded_file = st.file_uploader("📤 Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

# 4️⃣ Prediction Section
if uploaded_file is not None:
    # Open and resize the uploaded image to match CIFAR-10 input size (32x32)
    img = Image.open(uploaded_file).resize((32, 32))
    
    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_container_width=False)

    # Convert image to numpy array and normalize pixel values (0-1)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    # Run prediction using the loaded model
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display results
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    st.write(f"**Predicted Class:** {classes[class_index]}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

else:
    st.info("⬆️ Please upload an image to start prediction.")

st.markdown("---")
st.caption("CIFAR-10 Image Classification App | Built with ❤️ using Streamlit and TensorFlow")
