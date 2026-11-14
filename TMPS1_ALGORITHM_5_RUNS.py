import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
def selective_nearest_neighbors(X, y, k=3, confidence_threshold=0.99):
    if len(X) <= k:
        return X, y
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


# -------------------- Full Pipeline (5 Runs) --------------------
def run_pipeline(file_path, label_count=101, n_runs=5, base_random_state=42):
    print(f"Φόρτωση dataset από: {file_path}")
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, :-label_count].values
    y = df.iloc[:, -label_count:].values.astype(int)

    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path)

    # Λίστες για την αποθήκευση των μετρήσεων από τα 5 runs
    run_thresholds = []
    run_base_hls = []
    run_final_hls = []
    run_rrs = []
    run_micro_f1s = []
    run_macro_f1s = []

    # ==========================================
    # ΒΡΟΧΟΣ ΤΩΝ 5 RUNS
    # ==========================================
    for run in range(n_runs):
        print(f"\n--- Εκκίνηση Run {run + 1} / {n_runs} ---")
        current_seed = base_random_state + run

        # Διαχωρισμός δεδομένων με διαφορετικό seed ανά run
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.33, random_state=current_seed)
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=current_seed
        )

        num_train_full = len(X_train_split)
        best_threshold = 0
        best_hl = float("inf")
        threshold_range = np.arange(0.50, 0.99, 0.01)

        thresholds_tried = []
        hamming_losses = []
        reduction_rates_tried = []

        # Tuning Loop: Εύρεση βέλτιστου threshold στο validation set
        for threshold in threshold_range:
            X_cleaned, y_cleaned = selective_nearest_neighbors(X_train_split, y_train_split,
                                                               confidence_threshold=threshold)
            if len(X_cleaned) == 0:
                continue

            hl, _, _ = evaluate_model(X_cleaned, y_cleaned, X_val, y_val)
            rr_t = 1.0 - (len(X_cleaned) / num_train_full)

            thresholds_tried.append(threshold)
            hamming_losses.append(hl)
            reduction_rates_tried.append(rr_t)

            if hl < best_hl:
                best_hl = hl
                best_threshold = threshold

        print(f"Run {run + 1} - Βέλτιστο t*: {best_threshold:.2f} (Val HL: {best_hl:.4f})")

        # Υπολογισμός Baseline για αυτό το run
        base_hl, _, _ = evaluate_model(X_temp, y_temp, X_test, y_test)

        # Εφαρμογή βέλτιστου threshold στο σύνολο εκπαίδευσης (X_temp)
        X_final_train, y_final_train = selective_nearest_neighbors(X_temp, y_temp, confidence_threshold=best_threshold)
        f_hl, f_micro, f_macro = evaluate_model(X_final_train, y_final_train, X_test, y_test)

        reduction = (1 - (len(X_final_train) / len(X_temp))) * 100

        # Αποθήκευση αποτελεσμάτων run στις λίστες
        run_thresholds.append(best_threshold)
        run_base_hls.append(base_hl)
        run_final_hls.append(f_hl)
        run_rrs.append(reduction)
        run_micro_f1s.append(f_micro)
        run_macro_f1s.append(f_macro)

        # ==========================================
        # ΔΗΜΙΟΥΡΓΙΑ ΣΥΝΘΕΤΟΥ ΓΡΑΦΗΜΑΤΟΣ (DUAL-AXIS)
        # ==========================================
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = 'dodgerblue'
        color2 = 'red'

        # Αριστερός Άξονας: Hamming Loss
        ax1.set_xlabel(r'Confidence Threshold ($\tau$)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Hamming Loss (HL)', color=color1, fontsize=11, fontweight='bold')
        ax1.plot(thresholds_tried, hamming_losses, marker='o', linestyle='-', color=color1, linewidth=1.5, markersize=6)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Προσθήκη περιθωρίου (margin) στον αριστερό άξονα για να μην κολλάει η γραμμή επάνω
        y_min_hl, y_max_hl = min(hamming_losses), max(hamming_losses)
        margin_hl = (y_max_hl - y_min_hl) * 0.15 if y_max_hl != y_min_hl else 0.01
        ax1.set_ylim(y_min_hl - margin_hl, y_max_hl + margin_hl * 2.5)

        # Δεξιός Άξονας: Reduction Rate
        ax2 = ax1.twinx()
        ax2.set_ylabel('Reduction Rate (RR)', color=color2, fontsize=11, fontweight='bold')
        ax2.plot(thresholds_tried, reduction_rates_tried, marker='s', linestyle='-', color=color2, linewidth=1.5,
                 markersize=6)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(-0.05, 1.15)  # Αφήνει χώρο στο επάνω μέρος για να χωράει η ετικέτα άνετα

        # Κάθετη γραμμή για το βέλτιστο threshold
        plt.axvline(x=best_threshold, color='black', linestyle='--', linewidth=1.2)

        # Εύρεση του RR στο βέλτιστο threshold
        best_idx = np.argmin(hamming_losses)
        best_rr_box = reduction_rates_tried[best_idx]

        # --- ΕΞΥΠΝΗ ΤΟΠΟΘΕΤΗΣΗ ΜΕ ΒΕΛΟΣ (ANNOTATE) ---
        # Υπολογισμός ασφαλούς θέσης για το πλαίσιο (αριστερά από το σημείο και ψηλά)
        # --- ΕΞΥΠΝΗ ΤΟΠΟΘΕΤΗΣΗ ΜΕ ΒΕΛΟΣ (ANNOTATE) ---
        text_x = best_threshold - 0.12 if best_threshold > 0.65 else 0.65
        text_y = 0.82  # Ύψος του πλαισίου κειμένου

        annotation_text = f"$\\tau^*$ = {best_threshold:.2f}\nHL = {best_hl:.4f}\nRR = {best_rr_box:.2f}"

        ax2.annotate(
            annotation_text,
            xy=(best_threshold, 0.95),  # Το βέλος δείχνει ψηλά πάνω στη διακεκομμένη γραμμή (95% του ύψους)
            xycoords=('data', 'axes fraction'),  # x σε τιμή threshold, y σε ποσοστό ύψους (0 έως 1)
            xytext=(text_x, text_y),  # Θέση του πλαισίου κειμένου
            textcoords='axes fraction',
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9),
            fontsize=10,
            fontweight='bold'
        )

        # Υπόμνημα (Legend)
        # Υπόμνημα (Legend) - Διορθωμένο με raw strings (r'...')
        legend_elements = [
            Line2D([0], [0], color=color1, marker='o', label=r'Validation HL ($\downarrow$)'),
            Line2D([0], [0], color=color2, marker='s', label=r'Reduction Rate ($\uparrow$)'),
            Line2D([0], [0], color='black', linestyle='--', label=f'Best $\\tau^*$ = {best_threshold:.2f}')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True, edgecolor='black')
        


        plt.title(f"Hamming Loss and Reduction Rate vs Threshold ({filename} - Run {run + 1})", fontsize=12,
                  fontweight='bold')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{filename}_run_{run + 1}_dual_axis_plot.png")
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)

    # ==========================================
    # ΤΕΛΟΣ ΤΩΝ RUNS - ΥΠΟΛΟΓΙΣΜΟΣ ΣΤΑΤΙΣΤΙΚΩΝ
    # ==========================================
    mean_t = np.mean(run_thresholds)
    std_t = np.std(run_thresholds, ddof=1)

    mean_base_hl = np.mean(run_base_hls)
    std_base_hl = np.std(run_base_hls, ddof=1)

    mean_final_hl = np.mean(run_final_hls)
    std_final_hl = np.std(run_final_hls, ddof=1)

    mean_rr = np.mean(run_rrs)
    std_rr = np.std(run_rrs, ddof=1)

    mean_micro = np.mean(run_micro_f1s)
    std_micro = np.std(run_micro_f1s, ddof=1)

    mean_macro = np.mean(run_macro_f1s)
    std_macro = np.std(run_macro_f1s, ddof=1)

    # Διαμόρφωση κειμένου αναφοράς
    summary_text = (
            "=" * 65 + "\n"
                       f"  ΣΥΓΚΕΝΤΡΩΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ TMPS1 (5 RUNS): {filename}\n"
                       "=" * 65 + "\n"
                                  f"Optimal Threshold (t*)    : {mean_t:.3f} ± {std_t:.3f}\n"
                                  f"Baseline Hamming Loss     : {mean_base_hl:.4f} ± {std_base_hl:.4f}\n"
                                  f"Final Hamming Loss (HL)   : {mean_final_hl:.4f} ± {std_final_hl:.4f}\n"
                                  f"Reduction Rate (RR)       : {mean_rr:.2f}% ± {std_rr:.2f}%\n"
                                  f"Micro-F1                  : {mean_micro:.4f} ± {std_micro:.4f}\n"
                                  f"Macro-F1                  : {mean_macro:.4f} ± {std_macro:.4f}\n"
                                  "=" * 65 + "\n"
    )

    print("\n" + summary_text)

    # ==========================================
    # ΑΠΟΘΗΚΕΥΣΗ ΣΤΟ ΑΡΧΕΙΟ (ΜΙΑ ΦΟΡΑ)
    # ==========================================
    output_txt_path = os.path.join(output_dir, f"{filename}_results_summary_tmps1.txt")

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"[SUCCESS] Το αρχείο αποθηκεύτηκε στο: {output_txt_path}")

    return {
        "threshold_mean": mean_t, "threshold_std": std_t,
        "hamming_loss_mean": mean_final_hl, "hamming_loss_std": std_final_hl,
        "reduction_rate_mean": mean_rr, "reduction_rate_std": std_rr
    }


# -------------------- Main Entry Point --------------------
if __name__ == "__main__":
    dataset_path = "/home/pit/PycharmProjects/TMPS1/mediamill_norm.csv"#CHANGE NAME OF DATASET
    results = run_pipeline(file_path=dataset_path, label_count=101, n_runs=5)#CHANGE COUNT OF LABELS
