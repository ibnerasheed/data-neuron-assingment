import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keras_tuner.tuners import RandomSearch

from keras_tuner.engine.hyperparameters import HyperParameters


df = pd.read_csv('./DataNeuron_Text_Similarity.csv')


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


model = SentenceTransformer('bert-base-nli-mean-tokens')


train_embeddings1 = model.encode(train_df['text1'].values)
train_embeddings2 = model.encode(train_df['text2'].values)


test_embeddings1 = model.encode(test_df['text1'].values)
test_embeddings2 = model.encode(test_df['text2'].values)


train_concatenated = np.concatenate([train_embeddings1, train_embeddings2], axis=1)
test_concatenated = np.concatenate([test_embeddings1, test_embeddings2], axis=1)


def build_siamese_model(hp):
    input1 = Input(shape=(768,))
    input2 = Input(shape=(768,))

    dense_layers = [Dense(512, activation='relu', name=f'dense_{i}') for i in range(hp.Int('num_layers', min_value=1, max_value=3))]
    processed1 = input1
    processed2 = input2
    for dense_layer in dense_layers:
        processed1 = dense_layer(processed1)
        processed2 = dense_layer(processed2)

    concatenated = Concatenate()([processed1, processed2])
    output_layer = Dense(train_concatenated.shape[1], name='output')(concatenated)

    siamese_model = Model(inputs=[input1, input2], outputs=output_layer)
    siamese_model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                          loss=MeanSquaredError(),
                          metrics=['accuracy'])
    return siamese_model





tuner = RandomSearch(
    build_siamese_model,
    objective='val_loss',
    max_trials= 10,  
    executions_per_trial=1,
    directory='model_training_params',
    project_name='siamese_hyperparam_tuning'
)


tuner.search([train_embeddings1, train_embeddings2], train_concatenated,
             validation_data=([test_embeddings1, test_embeddings2], test_concatenated),
             epochs=10,
             batch_size=32)


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"The best hyperparameters are: {best_hps}")


best_model = tuner.get_best_models(num_models=1)[0]


best_model.fit([train_embeddings1, train_embeddings2], train_concatenated,
               validation_data=([test_embeddings1, test_embeddings2], test_concatenated),
               epochs=10,
               batch_size=32)

best_model.save('best_siamese_model.keras')



test_concatenated_predictions = best_model.predict([test_embeddings1, test_embeddings2])
similarity_scores = cosine_similarity(test_concatenated_predictions[:, :768], test_concatenated_predictions[:, 768:])





