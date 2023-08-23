import pandas as pd
import os


def collect_data():
    """Collecte des données de l'utilisateur."""
    
    ext_source_3 = float(input("Enter value for EXT_SOURCE_3: "))
    ext_source_2 = float(input("Enter value for EXT_SOURCE_2: "))
    name_income_type_working = int(input("Enter value for NAME_INCOME_TYPE_Working (1 or 0): "))
    name_education_type_secondary = int(input("Enter value for NAME_EDUCATION_TYPE_Secondary / secondary special (1 or 0): "))
    name_education_type_higher = int(input("Enter value for NAME_EDUCATION_TYPE_Higher education (1 or 0): "))
    occupation_type_core_staff = int(input("Enter value for OCCUPATION_TYPE_Core staff (1 or 0): "))
    flag_document_3 = int(input("Enter value for FLAG_DOCUMENT_3 (1 or 0): "))
    amt_req_credit_bureau_hour = float(input("Enter value for AMT_REQ_CREDIT_BUREAU_HOUR: "))
    code_gender = int(input("Enter value for CODE_GENDER (1 or 0): "))
    
    montant_remboursement = float(input("Enter Montant du remboursement: "))
    montant_credit = float(input("Enter Montant du credit: "))
    
    payment_rate = montant_remboursement / montant_credit

    data = {
        "EXT_SOURCE_3": ext_source_3,
        "EXT_SOURCE_2": ext_source_2,
        "NAME_INCOME_TYPE_Working": name_income_type_working,
        "NAME_EDUCATION_TYPE_Secondary / secondary special": name_education_type_secondary,
        "NAME_EDUCATION_TYPE_Higher education": name_education_type_higher,
        "OCCUPATION_TYPE_Core staff": occupation_type_core_staff,
        "FLAG_DOCUMENT_3": flag_document_3,
        "AMT_REQ_CREDIT_BUREAU_HOUR": amt_req_credit_bureau_hour,
        "CODE_GENDER": code_gender,
        "PAYMENT_RATE": payment_rate
    }

    return data

def save_data(data):
    """Enregistre les données collectées dans un fichier CSV."""

    # Chemin du fichier
    file_path = "test_data.csv"

    # Lire les données existantes
    df = pd.read_csv(file_path, index_col=0)

    # Récupère le dernier index et l'incrémente
    last_index = df.index[-1]
    number = int(last_index[1:]) + 1
    new_index = "D" + str(number).zfill(6)

    # Ajoutez la nouvelle entrée avec le nouvel index
    df = df.append(pd.Series(data, name=new_index))

    # Enregistrez les données mises à jour
    df.to_csv(file_path)

if __name__ == "__main__":
    user_data = collect_data()
    save_data(user_data)