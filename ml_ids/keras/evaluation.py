PREDICT_BATCH_SIZE = 16384


def predict(model, X, decision_boundary=0.5):
    pred = model.predict(X, batch_size=PREDICT_BATCH_SIZE)
    return (pred >= decision_boundary).astype('int').reshape(-1)


def predict_score(model, X):
    return model.predict(X, batch_size=PREDICT_BATCH_SIZE).reshape(-1)


def evaluate_model(model, X_train, y_train, X_val, y_val):
    print('Evaluation:')
    print('===========')
    print('       Loss / PR AUC / Precision / Recall')
    print('Train: {}'.format(model.evaluate(X_train, y_train, batch_size=PREDICT_BATCH_SIZE, verbose=0)))
    print('Val:   {}'.format(model.evaluate(X_val, y_val, batch_size=PREDICT_BATCH_SIZE, verbose=0)))
