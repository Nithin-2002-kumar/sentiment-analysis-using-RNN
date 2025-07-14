import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters
max_features = 10000  # Number of unique words to consider
maxlen = 100          # Maximum review length after padding
embedding_dim = 32    # Size of word embeddings

# Load IMDB dataset
print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(f"Train sequences: {len(x_train)}, Test sequences: {len(x_test)}")
print("Example sequence:", x_train[0])

# Pad sequences to ensure uniform input shape
print("Padding sequences...")
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Build the RNN model
print("Building model...")
model = Sequential([
    Embedding(max_features, embedding_dim, input_length=maxlen),
    SimpleRNN(32),  # Use 32 hidden units in the RNN layer
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train the model
print("Training model...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Evaluate the model on test data
print("Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
