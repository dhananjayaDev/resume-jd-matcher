from model.trainer import train_regressor

if __name__ == "__main__":
    model, reg = train_regressor()
    print("Regressor trained and saved.")