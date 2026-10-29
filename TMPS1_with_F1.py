import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import hamming_loss, f1_score
from sklearn.multioutput import MultiOutputClassifier


# -------------------- Evaluation Function --------------------
def evaluate_model(X_train, y_train, X_test, y_test):
    model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=1))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    hl = hamming_loss(y_test, y_pred)
    micro = f1_score(y_test, y_pred, average='micro')
    macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    return hl, micro, macro


# -------------------- Selective Nearest Neighbors --------------------
def selective_nearest_neighbors(X, y, k=3, confidence_threshold=0.98):
    if len(X) <= k: return X, y
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    indices = knn.kneighbors(X, return_distance=False)[:, 1:]

    keep_indices = []
    for i, neighbors in enumerate(indices):
        hamming_distances = [hamming_loss(y[i], y[n]) for n in neighbors]
        avg_hamming = np.mean(hamming_distances)
        confidence = 1 - avg_hamming
        if confidence >= confidence_threshold:
            keep_indices.append(i)

    return X[keep_indices], y[keep_indices]


# -------------------- Full Pipeline --------------------
def run_pipeline(file_path, label_count=101, random_state=42):
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, :-label_count].values
    y = df.iloc[:, -label_count:].values.astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_temp, y_temp, test_size=0.25,
                                                                  random_state=random_state)

    best_threshold = 0
    best_hl = float("inf")
    threshold_range = np.arange(0.50, 0.981, 0.01)

    # Tuning Loop: Find the best threshold first
    for threshold in threshold_range:
        X_cleaned, y_cleaned = selective_nearest_neighbors(X_train_split, y_train_split, confidence_threshold=threshold)
        if len(X_cleaned) == 0: continue

        hl, _, _ = evaluate_model(X_cleaned, y_cleaned, X_val, y_val)
        if hl < best_hl:
            best_hl = hl
            best_threshold = threshold

    # --- NOW Step 2, 3, 4 are OUTSIDE the loop ---

    # Calculate Baseline (BRkNN on Full Data)
    base_hl, base_micro, base_macro = evaluate_model(X_temp, y_temp, X_test, y_test)

    # Calculate SNN Results (BRkNN on Cleaned Data using the best threshold found)
    X_final_train, y_final_train = selective_nearest_neighbors(X_temp, y_temp, confidence_threshold=best_threshold)
    f_hl, f_micro, f_macro = evaluate_model(X_final_train, y_final_train, X_test, y_test)

    reduction = (1 - (len(X_final_train) / len(X_temp))) * 100

    print("\n" + "=" * 55)
    print(f"RESULTS FOR: {os.path.basename(file_path)}")
    print(f"Best Threshold Found: {best_threshold:.2f}")
    print("=" * 55)
    print(f"{'METRIC':<20} | {'BASELINE (BRkNN)':<16} | {'SNN + BRkNN':<12}")
    print("-" * 55)
    print(f"{'Hamming Loss':<20} | {base_hl:<16.4f} | {f_hl:<12.4f}")
    print(f"{'Micro F1':<20} | {base_micro:<16.4f} | {f_micro:<12.4f}")
    print(f"{'Macro F1':<20} | {base_macro:<16.4f} | {f_macro:<12.4f}")
    print(f"{'Training Samples':<20} | {len(X_temp):<16} | {len(X_final_train):<12}")
    print(f"{'Reduction Rate':<20} | {'0.00%':<16} | {reduction:<11.2f}%")
    print("=" * 55)

    return f_hl, f_micro, f_macro


# -------------------- Main Entry Point --------------------
if __name__ == "__main__":
    # Adjust label_count according to your dataset (e.g., CAL500=174, emotions=6, yeast=14)
    dataset_path = "/home/pit/PycharmProjects/ML_SNN/mediamill_norm.csv"
    run_pipeline(file_path=dataset_path, label_count=101)
