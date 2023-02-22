from oscillator import Oscillator

from sdr.skl.sdr import SDRRegressor

pend = Oscillator(b=0.1, noise=0.05)
X = pend.X
y = X[0]
sdr = SDRRegressor()
sdr.fit(X, y)
sdr.predict(X)
sdr.present_results(X)

if __name__ == "__main__":
    ...
