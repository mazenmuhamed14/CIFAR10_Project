import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model(model_path='models/cifar10_cnn_model.h5', data_path='data'):
    x_test = np.load(os.path.join(data_path, 'x_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    model = load_model(model_path)
    print(f"📦 Model loaded from: {model_path}")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {acc*100:.2f}%")
    print(f"🔹 Test Loss: {loss:.4f}")
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
