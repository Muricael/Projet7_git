import pandas as pd
import numpy as np
from mes_fonctions import one_hot_encoder
from mes_fonctions import application_train_test
from mes_fonctions import bureau_and_balance
from mes_fonctions import previous_applications
from mes_fonctions import pos_cash
from mes_fonctions import installments_payments
from mes_fonctions import credit_card_balance

def test_one_hot_encoder():
    # Création d'un DataFrame exemple pour le test
    df = pd.DataFrame({
        'A': ['a', 'b', 'c', 'a', np.nan],
        'B': ['x', 'y', 'y', 'x', 'y'],
        'C': [1, 2, 3, 4, 5]
    })

    # Appliquez la fonction one_hot_encoder
    df_encoded, new_columns = one_hot_encoder(df, nan_as_category=True)

    # Assurez-vous que les nouvelles colonnes ont été créées
    assert 'A_a' in df_encoded
    assert 'A_b' in df_encoded
    assert 'A_c' in df_encoded
    assert 'A_nan' in df_encoded
    assert 'B_x' in df_encoded
    assert 'B_y' in df_encoded

    # Assurez-vous que les colonnes originales ont été supprimées
    assert 'A' not in df_encoded
    assert 'B' not in df_encoded

    # Vérifiez si la transformation a bien fonctionné pour une ligne
    assert df_encoded.loc[0, 'A_a'] == 1
    assert df_encoded.loc[0, 'B_x'] == 1

    # Testez si KNNImputer a bien fonctionné (dans ce cas simple, il n'y a pas de NaN pour la transformation)
    assert not df_encoded[new_columns].isnull().values.any()

def test_application_train_test():

    # Charger un petit échantillon de données pour le test
    df_test = application_train_test(num_rows=100)

    # 1. Test de chargement des données
    assert isinstance(df_test, pd.DataFrame), "Le résultat devrait être un DataFrame"
    assert not df_test.empty, "Le DataFrame ne devrait pas être vide"

    # 2. Test des valeurs manquantes
    assert df_test["DAYS_EMPLOYED"].isnull().sum() == 0, "Il devrait y avoir 0 valeurs manquantes dans DAYS_EMPLOYED"

    # 3. Test d'encodage
    assert "CODE_GENDER" in df_test.columns, "CODE_GENDER devrait exister"
    assert df_test["CODE_GENDER"].nunique() <= 2, "CODE_GENDER devrait avoir 2 valeurs uniques au maximum"
    
    # 4. Test des nouvelles caractéristiques
    new_features = ["DAYS_EMPLOYED_PERC", "INCOME_CREDIT_PERC", "INCOME_PER_PERSON", "ANNUITY_INCOME_PERC", "PAYMENT_RATE"]
    for feature in new_features:
        assert feature in df_test.columns, f"{feature} devrait être dans le DataFrame"
        assert df_test[feature].isnull().sum() == 0, f"{feature} ne devrait pas avoir de valeurs manquantes"

    # 5. Test de la taille du dataframe
    assert len(df_test) == 200, "La taille du DataFrame devrait être de 100"

    print("Tous les tests ont passé!")



def test_columns_have_expected_datatypes():
    df = bureau_and_balance(1000)
    for col in df.columns:
        # Vérifiez que le type de données de chaque colonne correspond à ce qui est attendu (numérique ici)
        assert pd.api.types.is_numeric_dtype(df[col]), f"La colonne {col} n'a pas un type de données numérique"

def test_no_negative_values_in_MONTHS_BALANCE_MIN():
    df = bureau_and_balance(1000)
    if "BURO_MONTHS_BALANCE_MIN" in df.columns:
        assert (df["BURO_MONTHS_BALANCE_MIN"] >= 0).all(), "Il y a des valeurs négatives dans BURO_MONTHS_BALANCE_MIN"

def test_aggregated_columns_present():
    df = bureau_and_balance(1000)
    # Vérifiez que certaines colonnes attendues sont présentes
    expected_cols = ["BURO_DAYS_CREDIT_MEAN", "BURO_DAYS_CREDIT_ENDDATE_MAX"]
    for col in expected_cols:
        assert col in df.columns, f"La colonne {col} est absente"

def test_positive_values_in_AMT_CREDIT_SUM():
    df = bureau_and_balance(1000)
    if "BURO_AMT_CREDIT_SUM_MEAN" in df.columns:
        assert (df["BURO_AMT_CREDIT_SUM_MEAN"] >= 0).all(), "Il y a des valeurs négatives dans BURO_AMT_CREDIT_SUM_MEAN"

# Vérifier qu'après agrégation, les valeurs de la colonne CREDIT_ACTIVE_Active sont entre 0 et 1 (après one hot encoding)
def test_credit_active_active_values():
    df = bureau_and_balance(1000)
    if "BURO_CREDIT_ACTIVE_Active_MEAN" in df.columns:
        assert ((df["BURO_CREDIT_ACTIVE_Active_MEAN"] >= 0) & (df["BURO_CREDIT_ACTIVE_Active_MEAN"] <= 1)).all(), "Les valeurs dans BURO_CREDIT_ACTIVE_Active_MEAN ne sont pas entre 0 et 1"

# Test pour vérifier qu'il n'y a pas de doublons
def test_no_duplicates():
    df = bureau_and_balance(1000)
    assert not df.duplicated().any(), "Il y a des lignes dupliquées dans le DataFrame résultant"

# Test pour s'assurer que la taille de la sortie est celle attendue
def test_expected_row_count():
    sample_size = 1000
    df = bureau_and_balance(sample_size)
    assert len(df) <= sample_size, "Le nombre de lignes dépasse l'échantillon d'entrée"

def test_previous_applications():

    # 1. Test de chargement des données
    df_prev = previous_applications(num_rows=100)
    assert isinstance(df_prev, pd.DataFrame), "Le résultat devrait être un DataFrame"
    assert not df_prev.empty, "Le DataFrame ne devrait pas être vide"


    # 2. Test d'encodage
    new_features = [
        "PREV_AMT_ANNUITY_MIN", "PREV_AMT_APPLICATION_MAX",
        "APPROVED_AMT_CREDIT_MEAN", "REFUSED_AMT_ANNUITY_MAX"
    ]
    for feature in new_features:
        assert feature in df_prev.columns, f"{feature} devrait être dans le DataFrame"

    # 3. Vérification des colonnes générées par one-hot encoding
    encoded_cols = ["NAME_CONTRACT_STATUS_Approved", "NAME_CONTRACT_STATUS_Refused"]
    for col in encoded_cols:
        assert col not in df_prev.columns, f"{col} ne devrait pas être dans le DataFrame"

    print("Tous les tests pour previous_applications ont passé!")

def test_pos_cash():

    # 1. Test de chargement des données
    df_pos = pos_cash(num_rows=100)
    assert isinstance(df_pos, pd.DataFrame), "Le résultat devrait être un DataFrame"
    assert not df_pos.empty, "Le DataFrame ne devrait pas être vide"

    # 2. Test d'encodage
    new_features = [
        "POS_MONTHS_BALANCE_MAX", "POS_SK_DPD_MEAN", "POS_SK_DPD_DEF_MAX", "POS_COUNT"
    ]
    for feature in new_features:
        assert feature in df_pos.columns, f"{feature} devrait être dans le DataFrame"
        assert df_pos[feature].isnull().sum() == 0, f"{feature} ne devrait pas avoir de valeurs manquantes"

    # 4. Vérification des colonnes générées par one-hot encoding
    original_columns = ["NAME_CONTRACT_STATUS"]
    for col in original_columns:
        assert col not in df_pos.columns, f"{col} ne devrait pas être dans le DataFrame"

    # 5. Vérification de la cohérence des données
    assert (df_pos["POS_COUNT"] > 0).all(), "POS_COUNT devrait toujours être positif"

    print("Tous les tests pour pos_cash ont passé!")

def test_installments_payments():

    df_ins = installments_payments(num_rows=100)
    assert isinstance(df_ins, pd.DataFrame), "Le résultat devrait être un DataFrame"
    assert not df_ins.empty, "Le DataFrame ne devrait pas être vide"


    new_features = ["INSTAL_PAYMENT_PERC_MAX", "INSTAL_DPD_MEAN", "INSTAL_DBD_SUM", "INSTAL_COUNT"]
    for feature in new_features:
        assert feature in df_ins.columns, f"{feature} devrait être dans le DataFrame"
        assert df_ins[feature].isnull().sum() == 0, f"{feature} ne devrait pas avoir de valeurs manquantes"

    original_columns = ["NUM_INSTALMENT_VERSION"]
    for col in original_columns:
        assert col not in df_ins.columns, f"{col} ne devrait pas être dans le DataFrame"

    assert (df_ins["INSTAL_COUNT"] > 0).all(), "INSTAL_COUNT devrait toujours être positif"

    print("Tous les tests pour installments_payments ont passé!")

def test_credit_card_balance():

    df_cc = credit_card_balance(num_rows=100)
    assert isinstance(df_cc, pd.DataFrame), "Le résultat devrait être un DataFrame"
    assert not df_cc.empty, "Le DataFrame ne devrait pas être vide"


    expected_features = ["CC_AMT_BALANCE_MAX", "CC_AMT_BALANCE_MEAN", "CC_CNT_INSTALMENT_MATURE_CUM_VAR", "CC_COUNT"]
    for feature in expected_features:
        assert feature in df_cc.columns, f"{feature} devrait être dans le DataFrame"

    original_columns = ["SK_ID_PREV"]
    for col in original_columns:
        assert col not in df_cc.columns, f"{col} ne devrait pas être dans le DataFrame"

    assert (df_cc["CC_COUNT"] > 0).all(), "CC_COUNT devrait toujours être positif"

    print("Tous les tests pour credit_card_balance ont passé!")