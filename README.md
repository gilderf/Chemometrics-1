# Chemometrics
a python package for chemometrics

# a simple example
- a PLS example from sklearn
~~~~
  from sklearn.cross_decomposition import PLSRegression
  X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
  Y = [.2, 1.1, 5.9,12.3]
  pls = PLSRegression(n_components=2)
  pls.fit(X, Y)
  Y_pred = pls.predict(X)
~~~~
# dependency
- numpy
- sklearn
- pandas
