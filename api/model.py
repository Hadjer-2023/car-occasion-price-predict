
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

# load dataset 
iris = load_iris()

# features && target
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train model
model = LogisticRegression(multi_class="auto", max_iter=100)
model.fit(X_train, y_train)

# score
model.score(X_train, y_train)

# predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print()
print(' > Predictions')
print(y_pred)
print()
print(' > Accuracy: ' + str(accuracy))
print()

dump(model, 'model.joblib')

print(' > Model saved successfull')
print()