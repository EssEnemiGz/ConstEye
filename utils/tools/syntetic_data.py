import numpy as np
import os

os.makedirs("data/curves", exist_ok=True)

def generate_synthetic_curves(class_label, n=50, length=2000):
    for i in range(n):
        flux = np.random.normal(1.0, 0.002, length)
        num_transits = np.random.randint(1, 4) if class_label != 0 else 0
        for _ in range(num_transits):
            transit_pos = np.random.randint(300, 1700)
            depth = np.random.uniform(0.001, 0.01) if class_label == 1 else np.random.uniform(0.0005, 0.005)
            width = np.random.randint(5, 50)
            flux[transit_pos:transit_pos+width] -= depth

        if class_label == 2:
            flux += np.random.normal(0, 0.001, length)

        filename = f"data/curves/synthetic_{class_label}_{i}.npz"
        np.savez(filename, flux=flux.astype(np.float32), label=np.array(class_label).astype(np.long))

generate_synthetic_curves(0, 2000)
generate_synthetic_curves(1, 2000)
generate_synthetic_curves(2, 2000)
