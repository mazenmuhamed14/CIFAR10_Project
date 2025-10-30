import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(save_path='data'):
    # إنشاء مجلد data لو مش موجود
    os.makedirs(save_path, exist_ok=True)

    print("📥 Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize القيم بين 0 و 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # تحويل الـ labels لـ one-hot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # حفظ البيانات في ملفات .npy داخل مجلد data
    np.save(os.path.join(save_path, 'x_train.npy'), x_train)
    np.save(os.path.join(save_path, 'y_train.npy'), y_train)
    np.save(os.path.join(save_path, 'x_test.npy'), x_test)
    np.save(os.path.join(save_path, 'y_test.npy'), y_test)

    print(f"✅ Data saved successfully in '{save_path}' folder.")
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

    return x_train, y_train, x_test, y_test

# لتجربته مباشرة:
if __name__ == "__main__":
    load_and_preprocess_data('data')
