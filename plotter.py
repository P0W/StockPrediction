import matplotlib.pyplot as plt
import numpy as np

def loadFile(fileName):
    result = []
    with open(fileName) as fh:
        for val in fh:
            result.append(float(val))

    return np.array(result)

closing_price_pred = loadFile('y_pred.txt')
closing_price_train = loadFile('y_train.txt')

plt.plot(closing_price_pred, label="Preds", marker="o")
plt.plot(closing_price_train, label="Data")
plt.legend()
plt.show()
