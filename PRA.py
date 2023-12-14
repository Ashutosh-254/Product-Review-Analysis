#Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score

#Loading the dataset 
data = pd.read_csv('amazon_reviews.csv')

#Column Names
X = data['Text']
y = data['Sentiment']

#Converting labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Set the size for Testing Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Tokenize the text data
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

#Convert text to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

#Pad sequences to ensure equal length for input to the neural network
max_sequence_length = 100
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

#Create a neural network for multi-class classification
num_classes = len(label_encoder.classes_)
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=100, input_length=max_sequence_length))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Train the model
model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.1)
#Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Accuracy: {accuracy:.2f}')

#Make predictions on the test set
predictions = model.predict_classes(X_test_padded)

#Display classification report
print(classification_report(y_test, predictions))
