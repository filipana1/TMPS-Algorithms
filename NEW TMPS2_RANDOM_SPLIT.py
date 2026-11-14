import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import hamming_loss, f1_score


def run_pipeline(file_path, label_count, n_runs=5):
    # ------------------------------------------
    # Φόρτωση Dataset & Διαχωρισμός Χαρακτηριστικών/Κλάσεων
    # ------------------------------------------
    print(f"Φόρτωση dataset από: {file_path}")
    df = pd.read_csv(file_path)

    X_df = df.iloc[:, :-label_count]
    Y_df = df.iloc[:, -label_count:]

    print(f"Συνολικά δείγματα: {df.shape[0]}")
    print(f"Στήλες γνωρισμάτων (Features): {X_df.shape[1]}")
    print(f"Στήλες κλάσεων (Labels): {Y_df.shape[1]}")

    # Λίστες για την αποθήκευση των αποτελεσμάτων από τα 5 runs
    run_thresholds = []
    run_hls = []
    run_rrs = []
    run_micro_f1s = []
    run_macro_f1s = []

    for run in range(n_runs):
        print(f"\n--- Εκκίνηση Run {run + 1} / {n_runs} ---")
        current_seed = 42 + run  # Διαφορετικό seed σε κάθε run για τυχαίο split

        # ------------------------------------------
        # 1. Διαχωρισμός Δεδομένων (Train/Val/Test)
        # ------------------------------------------
        X_train_full, X_test, Y_train_full, Y_test = train_test_split(
            X_df, Y_df, test_size=0.33, random_state=current_seed
        )

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_full, Y_train_full, test_size=0.25, random_state=current_seed
        )

        # ------------------------------------------
        # 2 & 3. Εντοπισμός Γειτόνων & Δείκτης Εμπιστοσύνης
        # ------------------------------------------
        k_neighbors = 3
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
        nn.fit(X_train)

        _, indices = nn.kneighbors(X_train)
        indices = indices[:, 1:]

        confidence_scores = []
        Y_train_np = Y_train.values

        for i in range(len(Y_train_np)):
            xi_labels = Y_train_np[i]
            neighbor_indices = indices[i]
            neighbor_labels = Y_train_np[neighbor_indices]

            hl_list = [hamming_loss(xi_labels, n_lbl) for n_lbl in neighbor_labels]
            mean_hl = np.mean(hl_list)

            conf_score = 1.0 - mean_hl
            confidence_scores.append(conf_score)

        confidence_scores = np.array(confidence_scores)

        # ------------------------------------------
        # 4. Επαναληπτική Βελτιστοποίηση Κατωφλίου (Grid Search & Skyline)
        # ------------------------------------------
        thresholds = np.round(np.arange(0.50, 0.99, 0.01), 2)
        results = []

        for t in thresholds:
            mask = confidence_scores >= t
            if np.sum(mask) == 0:
                continue

            X_filtered = X_train.iloc[mask]
            Y_filtered = Y_train.iloc[mask]

            rr_t = 1.0 - (len(X_filtered) / len(X_train))

            brknn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
            brknn.fit(X_filtered, Y_filtered)

            Y_val_pred = brknn.predict(X_val)
            hl_t = hamming_loss(Y_val, Y_val_pred)

            results.append({'threshold': t, 'HL': hl_t, 'RR': rr_t, '1-RR': 1.0 - rr_t})

        results_df = pd.DataFrame(results)

        if results_df.empty:
            print(f"Προειδοποίηση: Δεν βρέθηκαν έγκυρα thresholds στο run {run + 1}. Συνέχεια...")
            continue

        def get_skyline(df_res):
            is_efficient = np.ones(df_res.shape[0], dtype=bool)
            hl_vals = df_res['HL'].values
            rr_vals = df_res['RR'].values

            for i in range(len(df_res)):
                is_efficient[i] = not np.any(
                    (hl_vals <= hl_vals[i]) & (rr_vals >= rr_vals[i]) &
                    ((hl_vals < hl_vals[i]) | (rr_vals > rr_vals[i]))
                )
            return df_res[is_efficient]

        skyline_df = get_skyline(results_df)

        # ------------------------------------------
        # ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ & ΥΠΟΛΟΓΙΣΜΟΣ ΑΠΟΣΤΑΣΗΣ d_T
        # ------------------------------------------
        hl_min, hl_max = results_df['HL'].min(), results_df['HL'].max()
        rr_min, rr_max = results_df['RR'].min(), results_df['RR'].max()

        hl_range = hl_max - hl_min if hl_max != hl_min else 1.0
        rr_range = rr_max - rr_min if rr_max != rr_min else 1.0

        skyline_df = skyline_df.copy()
        skyline_df['HL_norm'] = (skyline_df['HL'] - hl_min) / hl_range
        skyline_df['RR_norm'] = (rr_max - skyline_df['RR']) / rr_range

        skyline_df['d_T'] = np.sqrt(skyline_df['HL_norm'] ** 2 + skyline_df['RR_norm'] ** 2)

        best_row = skyline_df.loc[skyline_df['d_T'].idxmin()]
        best_threshold = best_row['threshold']

        print(f"Run {run + 1} - Βέλτιστο t*: {best_threshold} (Val HL: {best_row['HL']:.4f}, RR: {best_row['RR']:.4f})")

        # ------------------------------------------
        # ΑΠΟΘΗΚΕΥΣΗ SKYLINE ΩΣ ΕΙΚΟΝΑ
        # ------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.scatter(results_df['1-RR'], results_df['HL'], color='lightgray', label='All Thresholds', alpha=0.7)
        plt.scatter(skyline_df['1-RR'], skyline_df['HL'], color='blue', label='Skyline Points', s=50)
        plt.scatter(best_row['1-RR'], best_row['HL'], color='red', marker='*', s=200, label=f'Best t* = {best_threshold}')
        plt.scatter([0], [0], color='green', marker='X', s=150, label='Ideal Point (0,0)')

        plt.xlabel('1 - Reduction Rate (1 - RR)')
        plt.ylabel('Hamming Loss (HL)')
        plt.title(f'Skyline Trade-off (Run {run + 1})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        image_path = f"/home/pit/PycharmProjects/TMPS_SKYLINE_VERSION/CAL500_plot_run_{run + 1}.png"#ALLAGI
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

        # ------------------------------------------
        # 5. Τελικό Φιλτράρισμα & 6. Αξιολόγηση Test Set
        # ------------------------------------------
        final_mask = confidence_scores >= best_threshold
        X_cleaned = X_train.iloc[final_mask]
        Y_cleaned = Y_train.iloc[final_mask]

        final_model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        final_model.fit(X_cleaned, Y_cleaned)

        Y_test_pred = final_model.predict(X_test)

        final_hl = hamming_loss(Y_test, Y_test_pred)
        micro_f1 = f1_score(Y_test, Y_test_pred, average='micro', zero_division=0)
        macro_f1 = f1_score(Y_test, Y_test_pred, average='macro', zero_division=0)
        final_rr = 1.0 - (len(X_cleaned) / len(X_train))

        run_thresholds.append(best_threshold)
        run_hls.append(final_hl)
        run_rrs.append(final_rr)
        run_micro_f1s.append(micro_f1)
        run_macro_f1s.append(macro_f1)

    # ------------------------------------------
    # 7. ΥΠΟΛΟΓΙΣΜΟΣ, ΕΚΤΥΠΩΣΗ & ΑΠΟΘΗΚΕΥΣΗ ΣΕ TXT
    # ------------------------------------------
    mean_t = np.mean(run_thresholds)
    std_t = np.std(run_thresholds, ddof=1)

    mean_hl = np.mean(run_hls)
    std_hl = np.std(run_hls, ddof=1)

    mean_rr = np.mean(run_rrs) * 100
    std_rr = np.std(run_rrs, ddof=1) * 100

    mean_micro = np.mean(run_micro_f1s)
    std_micro = np.std(run_micro_f1s, ddof=1)

    mean_macro = np.mean(run_macro_f1s)
    std_macro = np.std(run_macro_f1s, ddof=1)

    summary_text = (
        "=" * 50 + "\n"
        "  ΣΥΓΚΕΝΤΡΩΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ (ΜΕΣΟΣ ΟΡΟΣ ± STDEV ΓΙΑ 5 RUNS)  \n"
        "=" * 50 + "\n"
        f"Dataset: {file_path}\n"
        f"Optimal Threshold (t*) : {mean_t:.3f} ± {std_t:.3f}\n"
        f"Hamming Loss (HL)      : {mean_hl:.4f} ± {std_hl:.4f}\n"
        f"Reduction Rate (RR)    : {mean_rr:.2f}% ± {std_rr:.2f}%\n"
        f"Micro-F1               : {mean_micro:.4f} ± {std_micro:.4f}\n"
        f"Macro-F1               : {mean_macro:.4f} ± {std_macro:.4f}\n"
        "=" * 50 + "\n"
    )

    print("\n" + summary_text)

    # Αποθήκευση στο αρχείο κειμένου
    output_txt_path = "/home/pit/PycharmProjects/TMPS_SKYLINE_VERSION/results_summary_CAL500.txt"#ALLAGI
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Τα αποτελέσματα αποθηκεύτηκαν επιτυχώς στο αρχείο: {output_txt_path}")

    return {
        't_mean': mean_t, 't_std': std_t,
        'hl_mean': mean_hl, 'hl_std': std_hl,
        'rr_mean': mean_rr, 'rr_std': std_rr,
        'micro_mean': mean_micro, 'micro_std': std_micro,
        'macro_mean': mean_macro, 'macro_std': std_macro
    }


# -------------------- Main Entry Point --------------------
if __name__ == "__main__":
    dataset_path = "/home/pit/PycharmProjects/TMPS_SKYLINE_VERSION/CAL500_norm.csv"#ALLAGI
    results = run_pipeline(file_path=dataset_path, label_count=174, n_runs=5)#ALLAGI