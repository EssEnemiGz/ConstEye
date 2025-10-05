import numpy as np
import pandas as pd

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
        
        df = pd.DataFrame({"flux": flux})
        df.to_csv(f"data/curves/synthetic_{class_label}_{i}.csv", index=False)

generate_synthetic_curves(0, 20)
generate_synthetic_curves(1, 20)
generate_synthetic_curves(2, 20)

