import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import hamming_loss
from sklearn.multioutput import MultiOutputClassifier

# -------------------- Hamming Loss Function --------------------
def compute_hamming_loss(y_true, y_pred):
    return hamming_loss(y_true, y_pred)

# -------------------- Selective Nearest Neighbors --------------------
def selective_nearest_neighbors(X, y, k=3, confidence_threshold=0.98):#FIND 3 NEAREST  NEIGHBORS
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    indices = knn.kneighbors(X, return_distance=False)[:, 1:]

    scores = []
    keep_indices = []

    for i, neighbors in enumerate(indices):
        hamming_distances = [compute_hamming_loss(y[i], y[n]) for n in neighbors]
        avg_hamming = np.mean(hamming_distances)
        confidence = 1 - avg_hamming
        if confidence >= confidence_threshold:
            keep_indices.append(i)
        scores.append(confidence)

    X_cleaned = X[keep_indices]
    y_cleaned = y[keep_indices]

    return X_cleaned, y_cleaned

# -------------------- BRkNN Implementation --------------------
def brknn_hamming_loss(X_train, y_train, X_val, y_val):
    model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=1))#BRKNN WITH K=1
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return compute_hamming_loss(y_val, y_pred)

# -------------------- Full Pipeline --------------------
def run_pipeline(file_path, label_count=101, random_state=42):#DATASET LABELS
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, :-label_count].values
    y = df.iloc[:, -label_count:].values.astype(int)

    # Step 1: Split into 67% train and 33% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

    # Step 2: Split temp into 75% train and 25% validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=random_state)

    # Step 3: Try different confidence thresholds
    best_threshold = 0
    best_hl = float("inf")
    threshold_range = np.arange(0.50, 0.981, 0.01)

    thresholds_tried = []
    hamming_losses = []

    for threshold in threshold_range:
        X_cleaned, y_cleaned = selective_nearest_neighbors(X_train_split, y_train_split, confidence_threshold=threshold)
        if len(X_cleaned) == 0:
            continue
        hl = brknn_hamming_loss(X_cleaned, y_cleaned, X_val, y_val)

        thresholds_tried.append(threshold)
        hamming_losses.append(hl)

        if hl < best_hl:
            best_hl = hl
            best_threshold = threshold

    # Step 4: Print best threshold
    print(f"Best Confidence Threshold: {best_threshold:.2f} with Hamming Loss: {best_hl:.4f}")

    # Step 5: Apply best threshold to the initial 67% train
    X_cleaned_final, y_cleaned_final = selective_nearest_neighbors(X_temp, y_temp, confidence_threshold=best_threshold)
    final_hl = brknn_hamming_loss(X_cleaned_final, y_cleaned_final, X_test, y_test)

    # Step 6: Print reduction rate and final performance
    reduction_rate = (1 - (len(X_cleaned_final) / len(X_temp))) * 100
    print(f"Final Data Reduction: {reduction_rate:.2f}%")
    print(f"Final Hamming Loss on Test Set: {final_hl:.4f}")

    # Step 7: Plot and save Hamming Loss vs Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_tried, hamming_losses, marker='o', linestyle='-', label='Hamming Loss')
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best = {best_threshold:.2f}')
    plt.scatter(best_threshold, best_hl, color='red')
    plt.title('Hamming Loss vs Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Hamming Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure with dataset name
    filename = os.path.splitext(os.path.basename(file_path))[0]
    plot_path = f"{filename}_hamming_loss_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Hamming Loss plot saved to: {plot_path}")

    return {
        "best_threshold": best_threshold,
        "validation_hamming_loss": best_hl,
        "reduction_rate": reduction_rate,
        "final_hamming_loss": final_hl
    }

# -------------------- Main Entry Point --------------------
if __name__ == "__main__":
    dataset_path = "/home/pit/PycharmProjects/ML_SNN/mediamill_norm.csv" #DATASET CSV FILE
    results = run_pipeline(file_path=dataset_path, label_count=101)#DATASET LABELS 

    print("\n📊 Final Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
