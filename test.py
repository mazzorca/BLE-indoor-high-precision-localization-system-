import utility

if __name__ == "__main__":
    datiReader1, datiTimeReader1 = utility.extract_and_apply_kalman_csv("dati3105run0r")
    datiReader2, datiTimeReader2 = utility.convertCSV_backup("dati3105run0r")
