import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import callbacks, layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from aslo_features import FEATURE_LEN, _fist_disambiguation_from_norm_coords


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

    num_cols = df.shape[1]
    # Check if dataset needs fist disambiguation feature enrichment (226 -> 252 features)
    if num_cols == 227:
        print("Enriching dataset with fist disambiguation feature geometry (226 -> 252 features)...")
        base_features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values

        h0_coords = base_features[:, 0:63]
        h1_coords = base_features[:, 63:126]

        h0_fist = np.array([_fist_disambiguation_from_norm_coords(r) for r in h0_coords], dtype=np.float32)
        h1_fist = np.array([_fist_disambiguation_from_norm_coords(r) for r in h1_coords], dtype=np.float32)

        enriched_features = np.hstack([base_features, h0_fist, h1_fist])

        if not os.path.exists("gesture_data_original_backup.csv"):
            df.to_csv("gesture_data_original_backup.csv", index=False)

        enriched_df = pd.DataFrame(enriched_features)
        enriched_df["label"] = labels
        enriched_df.to_csv(data_path, index=False)
        print(f"Saved enriched dataset with shape {enriched_df.shape} to '{data_path}'")
        X = enriched_features
        y = labels
    elif num_cols == 253:
        print(f"Dataset already enriched with 252 features ({df.shape[0]} rows).")
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    assert X.shape[1] == FEATURE_LEN, f"Expected {FEATURE_LEN} features, got {X.shape[1]}"

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    np.save("label_classes.npy", label_encoder.classes_)
    print(f"Found {num_classes} classes: {label_encoder.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Compute balanced class weights to combat 'S' dominance over closed fists
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Anti-'S' Dominance Penalization:
    # Heavily penalize the model if it lazily predicts 'S' for 'A', 'E', 'M', 'N', or 'T'
    fist_classes = {"A", "E", "M", "N", "T"}
    for idx, cls_name in enumerate(label_encoder.classes_):
        if cls_name in fist_classes and idx in class_weights:
            class_weights[idx] *= 1.35
        elif cls_name == "S" and idx in class_weights:
            class_weights[idx] *= 0.85

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

    model = models.Model(inp, out, name="GestureNet_v2_fist_stabilized")

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    )

    print("Starting model training with balanced class weighting...")
    history = model.fit(
        X_train,
        y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[early_stopping],
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(
        f"\nModel Evaluation - Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}"
    )

    model_path = "gesture_model.keras"
    model.save(model_path)
    print(f"Model saved successfully to '{model_path}'")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Output detailed precision & recall report for closed-fist gestures
    target_names = [str(c) for c in label_encoder.classes_]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    print("\n" + "=" * 62)
    print("FIST DISAMBIGUATION PRECISION / RECALL REPORT [A, E, M, N, S, T]")
    print("=" * 62)
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 56)
    for cls in ["A", "E", "M", "N", "S", "T"]:
        if cls in report:
            r = report[cls]
            print(f"{cls:<8} {r['precision']:<12.3f} {r['recall']:<12.3f} {r['f1-score']:<12.3f} {int(r['support']):<10}")
    print("=" * 62 + "\n")

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
