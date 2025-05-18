import scipy.io
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def load_data_features_labels(mat_path):
    mat = scipy.io.loadmat(mat_path)
    data = mat['data']  # shape: (n_trials, n_samples, n_channels)
    fs = mat['fs'][0][0]  # sampling freq scalar
    labels = mat['labels'][0]  # Assuming labels shape is (1, n_trials) or (n_trials,)

    n_trials, n_samples, n_channels = data.shape

    # Step 1: Common Average Reference (CAR)
    car_data = np.empty_like(data, dtype=np.float64)
    for i in range(n_trials):
        trial = data[i]
        avg_across_channels = np.mean(trial, axis=1, keepdims=True)
        car_data[i] = trial - avg_across_channels

    # Step 2: Compute power spectrum and band averages
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    trial_band_averages = []
    for trial in range(n_trials):
        power_sum = {band: 0 for band in bands}
        count = {band: 0 for band in bands}

        for ch in range(n_channels):
            signal = car_data[trial, :, ch]
            fft_result = np.fft.rfft(signal)
            power = np.abs(fft_result) ** 2 / n_samples

            for i, f in enumerate(freqs):
                for band, (low, high) in bands.items():
                    if low <= f < high:
                        power_sum[band] += power[i]
                        count[band] += 1

        band_avgs = {band: (power_sum[band] / count[band] if count[band] else 0) for band in bands}
        trial_band_averages.append(band_avgs)

    # Convert to feature matrix X
    X = []
    for i in range(n_trials):
        band_values = [trial_band_averages[i][band] for band in bands]
        X.append(band_values)  # No trial number column, just features
    X = np.array(X)

    return X, labels

# Load train features and labels
X_train, y_train = load_data_features_labels(r"C:\Users\hadig\OneDrive\Desktop\train_data_1.mat")
# Load test features and labels
X_test, y_test = load_data_features_labels(r"C:\Users\hadig\OneDrive\Desktop\test_data_1.mat")

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Try KNN with different k values
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"K = {k}: Accuracy = {acc:.4f}")
