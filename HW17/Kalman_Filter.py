import numpy as np
import matplotlib.pyplot as plt

sigma = 0.1
R = sigma * sigma

t = [0]

x = 0.4
y = [x + np.random.normal(0, sigma)]
x_hat = [y[0]]

P = 0.2
K = P / (P + R)

iterations = 1000

for i in range(1, iterations):
    t.append(i)
    y.append(x + np.random.normal(0, sigma))
    K = P / (P + R)
    x_hat.append(x_hat[-1] + K * (y[-1] - x_hat[-1]))
    P = (1 - K) * P

plt.subplot(121)
plt.scatter(t, y, color="darkgray", marker="+", label=r"$noisy measurement$")
plt.plot(t, [x for i in range(iterations)], color="green", label=r"$true value$")
plt.plot(t, x_hat, color="blue", label=r"$KF estimation$")
plt.legend()
plt.title("result")
plt.subplot(122)
plt.plot(t, [0 for i in range(iterations)], color="green", label=r"$zero$")
plt.plot(t, [x_hat[i] - x for i in range(iterations)], color="blue", linewidth=1, label=r"$errors$")
plt.legend()
plt.title("error estimate")
plt.show()