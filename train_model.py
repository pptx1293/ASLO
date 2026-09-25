import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import callbacks, layers, models, regularizers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    data_path = "gesture_data.csv"

    if not os.path.exists(data_path):
        print(
            f"Error: '{data_path}' not found. Please run collect_data.py first to generate the dataset."
        )
        return

    print(f"Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)

    if df.empty:
        print("Dataset is empty. Please ensure images were properly processed.")
        return

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    np.save("label_classes.npy", label_encoder.classes_)
    print(f"Found {num_classes} classes: {label_encoder.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    inp = layers.Input(shape=(X_train.shape[1],))
    x = layers.GaussianNoise(0.02)(inp)

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.0005))(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.0005))(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.0005))(
        x
    )
    x = layers.BatchNormalization()(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="GestureNet_v2")

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    )

    print("Starting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(
        f"\nModel Evaluation - Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}"
    )

    model_path = "gesture_model.keras"
    model.save(model_path)
    print(f"Model saved successfully to '{model_path}'")

    plt.figure(figsize=(14, 12))

    plt.subplot(2, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.subplot(2, 1, 2)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    report_path = "training_report.png"
    plt.tight_layout()
    plt.savefig(report_path)
    print(f"Training report saved to '{report_path}'")

    cm_path = "confusion_matrix.png"
    if os.path.exists(cm_path):
        try:
            os.remove(cm_path)
        except OSError:
            pass

    print("Training complete! You can now run app.py for real-time detection.")


if __name__ == "__main__":
    main()
