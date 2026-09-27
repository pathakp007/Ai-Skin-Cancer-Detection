from src.predict import predict_image

image_path = "test.jpg"   # Replace with your test image

disease, confidence = predict_image(image_path)

print("Prediction :", disease)
print("Confidence :", confidence)