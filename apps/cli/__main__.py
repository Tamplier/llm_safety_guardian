from src.scripts.model_load import predict_with_proba

while True:
    user_input = input("> ").strip()
    if not user_input:
        break
    predicted = predict_with_proba([user_input]).to_dict(orient='records')
    print(predicted[0])
