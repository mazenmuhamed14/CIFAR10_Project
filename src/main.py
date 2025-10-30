from preprocess import load_and_preprocess_data
from model import build_model, train_model
from evaluation import evaluate_model
from visualizations import plot_training_history

if __name__ == "__main__":
    print("📦 Step 1: Loading and preprocessing data...")
    x_train, y_train, x_test, y_test = load_and_preprocess_data('data')

    print("\n🧠 Step 2: Building model...")
    model = build_model()

    print("\n🚀 Step 3: Training model...")
    history = train_model(model, x_train, y_train, x_test, y_test, 'models')

    print("\n📊 Step 4: Visualizing training results...")
    plot_training_history(history)

    print("\n🧾 Step 5: Evaluating model performance...")
    evaluate_model('models/cifar10_cnn_model.h5', 'data')

    print("\n✅ All steps completed successfully!")
