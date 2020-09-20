import keras
from keras.models import Sequential
from keras.layers import Dense


# Method to create and train the model_cost of solution

def cost_of_solution(solution, epochs, x_train, y_train, x_test, y_test):
    nb_neurons, nb_layers, batch_size, optimizer, activation = solution

    model = Sequential()
    model.add(Dense(nb_neurons, input_dim=13, activation=activation))
    for i in range(nb_layers):
        model.add(Dense(nb_neurons, activation=activation))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    model.fit(x_train, y_train, epochs=epochs, shuffle=True, batch_size=batch_size, verbose=0)
    score, error = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    keras.backend.clear_session()
    return error
