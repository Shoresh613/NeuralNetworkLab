import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

values_direct = [
    105,
    120,
    50,
    210,
    280,
    185,
    80,
    110,
    105,
    105,
    275,
    90,
    285,
    160,
    355,
    210,
    490,
    225,
    110,
    210,
    110,
    35,
    95,
    65,
    65,
    170,
    335,
    60,
    210,
    10,
    265,
    225,
    225,
    210,
    135,
    135,
    230,
    35,
    35,
    285,
    45,
    75,
    60,
    215,
    125,
    90,
    170,
    20,
    50,
    140,
    15,
    125,
    280,
    50,
    120,
    110,
    135,
    45,
    55,
    120,
    155,
    50,
    110,
    440,
    455,
    535,
    110,
    75,
    170,
    90,
    410,
    120,
    230,
    255,
    125,
    170,
    210,
    210,
    80,
    345,
    105,
    210,
    285,
    150,
    215,
    150,
    170,
    75,
    155,
    360,
    155,
    375,
    110,
    180,
    135,
    145,
    195,
    110,
    210,
    60,
    55,
]

indices_direct = np.arange(len(values_direct))

slope_direct, intercept_direct, r_value_direct, _, _ = linregress(
    indices_direct, values_direct
)

regression_line_direct = slope_direct * indices_direct + intercept_direct
avg_direct = np.mean(values_direct) * np.ones_like(indices_direct)

plt.figure(figsize=(10, 6))
plt.scatter(indices_direct, values_direct, alpha=0.7, label="Actual Data")
plt.plot(
    indices_direct,
    regression_line_direct,
    color="red",
    label=f"Trend (slope={slope_direct:.4f})",
)
plt.plot(
    indices_direct,
    avg_direct,
    color="blue",
    label=f"Average ({avg_direct[0]:.1f})",
)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("100 Episodes w/ Batch Normalization and Additional Layers")
plt.legend()
plt.show()

print(f"Slope: {slope_direct}, R²: {r_value_direct**2}")
