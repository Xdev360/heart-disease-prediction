from app.models.model_training import train_and_evaluate_model

if __name__ == "__main__":
    model, metrics = train_and_evaluate_model()
    print("Training complete. Metrics:", metrics) 