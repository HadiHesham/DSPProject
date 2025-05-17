import scipy.io
import numpy as np

# Load your .mat file
mat = scipy.io.loadmat(r"C:\Users\hadig\OneDrive\Desktop\test_data_1.mat")

# Explore keys (you already did)
print(mat.keys())

# Access actual data
data = mat['data']       # EEG data
labels = mat['labels']   # Corresponding labels
fs = mat['fs']           # Sampling frequency
channels = mat['channels']  # Optional: electrode names

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
print("Sampling frequency:", fs)
print(channels)
print(len(labels[0]))
print(len(data[0]))

new = []
for i in range(len(data)):
    trial = []
    for j in range(len(data[i])):
        sum = 0;
        for k in range(len(data[i][j])):
            sum = sum + data[i][j][k]
        average = sum/len(data[i][j])
        row = []
        for k in range(len(data[i][j])):
            row.append(data[i][j][k] - average)
        trial.append(row)
    new.append(trial)
print(new[0][0])

n_trials, n_samples, n_channels = data.shape

# Frequency axis
freqs = np.fft.rfftfreq(n_samples, d=1 / fs[0][0])  # Only non-negative frequencies

# Initialize power spectrum
power = np.zeros((n_trials, n_channels, len(freqs)))

# Loop through trials and channels
for trial in range(n_trials):
    for ch in range(n_channels):
        signal = data[trial, :, ch]
        fft_result = np.fft.rfft(signal)  # 1D FFT
        power[trial, ch, :] = np.abs(fft_result) ** 2 / n_samples  # Power spectrum

power_1d = power[0, 0, :]
print("Frequencies:", len(freqs))
print("Power shape:", len(power_1d))

delta = []
theta = []
alpha = []
beta = []
gamma = []
i=0
for f in freqs:
    f = float(f)
    row = [f, float(power_1d[i])]
    if f >= 0.5 and f<4:
        delta.append(row)
    if f >= 4 and f<8:
        theta.append(row)
    if f >= 8 and f<13:
        alpha.append(row)
    if f >= 13 and f<30:
        beta.append(row)
    if f >= 30 and f<100:
        gamma.append(row)
    i=i+1
print("Delta:", delta)
print("Theta:", theta)
print("Alpha:", alpha)
print("Beta:", beta)
print("Gamma:", gamma)

sum = 0
for pwrs in delta:
    sum = sum + pwrs[1]
AVGDelta = sum/len(delta)
print("AVGDelta:", AVGDelta)

sum = 0
for pwrs in theta:
    sum = sum + pwrs[1]
AVGTheta = sum/len(theta)
print("AVGTheta:", AVGTheta)

sum = 0
for pwrs in alpha:
    sum = sum + pwrs[1]
AVGAlpha = sum/len(alpha)
print("AVGAlpha:", AVGAlpha)

sum = 0
for pwrs in beta:
    sum = sum + pwrs[1]
AVGBeta = sum/len(beta)
print("AVGBeta:", AVGBeta)

sum = 0
for pwrs in gamma:
    sum = sum + pwrs[1]
AVGGamma = sum/len(gamma)
print("AVGGamma:", AVGGamma)