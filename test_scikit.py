import pytest
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import datasets
from joblib import dump, load


class TestClass():


    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_x_train = diabetes_X[:-20]
    diabetes_x_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    test_model = LinearRegression()
    test_model.fit(diabetes_x_train, diabetes_y_train)
    y_pred_test = test_model.predict(diabetes_x_test)


    # Test_ID NF-2
    def test_save_and_load_model(self):
        # Create linear regression object
        model = LinearRegression()
        model.fit(self.diabetes_x_train, self.diabetes_y_train)
        # Save and load model using joblib
        dump(model, "lin_model_test.joblib")
        model2 = load("lin_model_test.joblib")

        diabetes_y_pred = model.predict(self.diabetes_x_test)
        load_y_pred = model2.predict(self.diabetes_x_test)

        assert diabetes_y_pred.all() == load_y_pred.all()
    
    
    # Test_ID NF-3
    @pytest.mark.parametrize('execution_number', range(20))
    def test_reliability(self, execution_number):
        model = LinearRegression()
        model.fit(self.diabetes_x_train, self.diabetes_y_train)
        diabetes_y_pred = model.predict((self.diabetes_x_test))

        # Compare every element
        assert diabetes_y_pred.all() == self.y_pred_test.all()
    
    
    # Test_ID NF-9
    @pytest.mark.parametrize('n_set', range(150))
    def test_volume_lin_reg(self, n_set):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

        # The bigger the n-value gets the longer it takes
        # But the results should stay exactly the same
        n = n_set
        print(n)
        for i in range(3, n):
            for j in range(3, n):
                X = np.append(X, [[i, j]], axis=0)

        Y = np.dot(X, np.array([1, 2])) + 3
        model = LinearRegression()
        model.fit(X, Y)

        assert model.score(X, Y) == 1.0


