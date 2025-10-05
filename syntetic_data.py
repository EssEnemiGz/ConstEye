import numpy as np

def generate_synthetic_curves(class_label, n=20, length=2000):
    for i in range(n):
        flux = np.random.normal(1.0, 0.002, length)
        if class_label == 1:
            transit_pos = np.random.randint(300, 1700)
            depth = np.random.uniform(0.001, 0.01)
            width = np.random.randint(10, 50)
            flux[transit_pos:transit_pos+width] -= depth
        elif class_label == 2:
            transit_pos = np.random.randint(300, 1700)
            depth = np.random.uniform(0.0005, 0.005)
            width = np.random.randint(5, 40)
            flux[transit_pos:transit_pos+width] -= depth
            flux += np.random.normal(0, 0.001, length)

        output_filename = f"data/curves/synthetic_{class_label}_{i}.npz"
        np.savez(output_filename, flux=flux.astype(np.float32), label=np.array(class_label).astype(np.long))

# Genera tus nuevos archivos en formato .npz
generate_synthetic_curves(0, 200)
generate_synthetic_curves(1, 200)
generate_synthetic_curves(2, 200)
