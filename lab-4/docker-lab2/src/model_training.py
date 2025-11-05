import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

if __name__ == '__main__': 
    # Load the Wine dataset
    wine = datasets.load_wine()
    X, y = wine.data, wine.target
    
    # Print dataset info
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature names: {wine.feature_names}")
    print(f"Target names: {wine.target_names}")
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Save the scaler for use in prediction
    import joblib
    joblib.dump(sc, 'scaler.pkl')
    
    # Build a more complex TensorFlow model for wine dataset (13 features)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, input_shape=(13,), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train with more epochs and early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, 
                        epochs=100, 
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop],
                        verbose=1)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save the model
    model.save('wine_model.keras')
    print("Model was trained and saved as 'wine_model.keras'")