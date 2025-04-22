import matplotlib.pyplot as plt #pour plotter les graphes 
from sklearn import svm, datasets #bibliothèque sklearn du machine learning
import numpy as np
import pandas as pd
import seaborn as sns
import os

try:
    # Spécifiez le chemin du répertoire où se trouve le fichier iris.csv
    chemin_repertoire = "C:\\Ton\\chemin\\d'accès\\"

    # Changer le répertoire de travail
    os.chdir(chemin_repertoire)
    print("Répertoire de travail actuel : ", os.getcwd())

    # Vérification de l'existence du fichier
    if not os.path.isfile("iris.csv"):
        raise FileNotFoundError("Le fichier 'iris.csv' n'a pas été trouvé dans le répertoire de travail actuel.")

    # Chargement du dataset iris à partir du fichier CSV
    data = pd.read_csv("iris.csv")

    # Affichage des 10 premières lignes du dataset 
    data.head(10)

    # Informations sur le dataset iris
    data.info()

    # Classification SVM 3 classes avec deux attributs
    from sklearn.model_selection import train_test_split

    # On définit les colonnes des trois classes 
    svm1 = data.replace({"species": {"setosa":1,"versicolor":2, "virginica":3}})

    # On spécifie les variables contenant les caractéristiques et les valeurs à prédire
    X = svm1.drop(['species','sepal_width','petal_width'],axis=1)
    Y = svm1['species']

    # On split le dataset sur deux 20% pour le test et 80% pour le training
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.20)

    # Model training
    from sklearn.svm import SVC
    svm = SVC()

    # Classification
    clf = svm.fit(X_train1, Y_train1)

    # Prédiction 
    pred = svm.predict(X_test1)

    # Score de la prédiction effectuée sur le dataset 
    print("Score de la prédiction: ", svm.score(X_test1, Y_test1))

    # Fonction pour définir une grille d'affichage
    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    # Fonction pour affichage des contours
    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    # On définit la figure et les axes 
    figure, axis = plt.subplots()

    # Titre pour les plots
    title = ('SVM classification 3 classes avec deux attributs sur iris')

    # On indique les labels des axes
    X0, X1 = X["sepal_length"], X["petal_length"]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(axis, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

    inv_name_dict = {"setosa":1,"versicolor":2, "virginica":3}
    colors = [inv_name_dict[item] for item in data['species']]
    axis.scatter(X0, X1, c=colors, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    axis.set_ylabel('Sepal length feature')
    axis.set_xlabel('Petal length feature')
    axis.set_xticks(())
    axis.set_yticks(())
    axis.set_title(title)
    plt.show()

    # Les tests de performances 
    from sklearn.metrics import classification_report, confusion_matrix
    print("Precision, Recall, Confusion matrix, in training\n")

    # Rapport de classification avec les metrics precision, recall, f1-score
    print(classification_report(Y_test1, pred))
    print(confusion_matrix(Y_test1, pred))

    # SVM classification avec 2 classes et deux attributs 
    svm2 = data.replace({"species": {"setosa":1,"versicolor":2, "virginica":3}})
    index = svm2[(svm2['species'] == 1)].index
    svm2.drop(index, inplace=True)
    svm2.head(10)

    X = svm2.drop(['species','sepal_width','petal_width'], axis=1)
    Y = svm2['species']
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.20)

    svm = SVC()
    clf = svm.fit(X_train2, Y_train2) 
    pred = svm.predict(X_test2)
    print("Taux de précision du modèle: ", svm.score(X_test2, Y_test2))

    # Plotting 
    figure, axis = plt.subplots()
    title = ('SVM classification sur iris 2 classes et deux attributs')
    X0, X1 = X["sepal_length"], X["petal_length"]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(axis, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    axis.scatter(X0, X1, c="yellow", cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    axis.set_ylabel('Sepal length')
    axis.set_xlabel('Petal length')
    axis.set_xticks(())
    axis.set_yticks(())
    axis.set_title(title)
    plt.show()

    print("Precision, Recall, Confusion matrix, in training\n")
    print(classification_report(Y_test2, pred))
    print(confusion_matrix(Y_test2, pred))

    # Classification SVM de deux classes avec quatre attributs
    svm3 = data.replace({"species": {"setosa":1,"versicolor":2,"virginica":3}})
    act = data[(data['species'] == 1)].index
    data.drop(act, inplace=True)
    data.head(10)

    X = data.drop(['species'], axis=1)
    Y = data['species']
    X.head(10)

    # Entrainer le modèle et effectuer la classification SVM 
    X_train, X_test, Y_train, Y_testo = train_test_split(X, Y, test_size=0.20)

    svm = SVC()
    clf = svm.fit(X_train, Y_train) 
    predi = svm.predict(X_test)
    print("Taux de précision du modèle: ", svm.score(X_test, Y_testo))

except Exception as e:
    print(f"Une erreur s'est produite : {e}")

# Pause pour empêcher la fermeture automatique
input("Appuyez sur Entrée pour fermer...")
