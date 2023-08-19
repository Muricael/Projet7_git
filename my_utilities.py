# my_utilities.py
from sklearn.metrics import confusion_matrix, make_scorer

def custom_loss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fn_weight = 10.0  # poids pour les faux négatifs
    fp_weight = 10.0  # poids pour les faux positifs
    return cm[0, 1] * fp_weight + cm[1, 0] * fn_weight

# Faire une fonction d'évaluation personnalisée pour GridSearch
custom_scorer = make_scorer(custom_loss, greater_is_better=False)
