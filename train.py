# Step 5: Train the model
history = model.fit(train_generator, epochs=10, validation_data=test_generator)
