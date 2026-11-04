"""
Обучение классификатора на датасете Iris.
Используется для практики работы с Git.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def main():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)          # ← вернули на место
    print(f"Accuracy на тесте: {score:.2%}\n")

    y_pred = model.predict(X_test)

    print("=== Матрица ошибок (строки — истина, столбцы — предсказание) ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Отчёт по метрикам ===")
    print(classification_report(y_test, y_pred, target_names=load_iris().target_names))


if __name__ == "__main__":
    main()