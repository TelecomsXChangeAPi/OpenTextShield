import fasttext

# Load the trained model
model = fasttext.load_model('ots_sms_model_v1.1.bin')

# Example SMS message
# message = "Your account has been temporarily locked out now. Please log in to https://tinyurlw22222222.com/bank verify your identity." ## phishing
# message ="(9bQqme34005Uq)"
# message = "Your Microsoft verfication code is 667982"
# message = "PayPal: 441999 is your security code. Don't share your code"
# message = "Federal Credit Union ALERT: Your Credit Card has been temporarily LOCKED. Please call Services line (954) 240-5411" ## spam
# message = "Apple Notification. Your Apple iCloud ID expires today. Log in to prevent deletion http://notapple.com/user-auth/online" ## spam
# message = "No i was trying to lock down the deal yesterday :)"
message = "Hi, Your birthday is coming soon. It is time to buy a 1 Cr Term Plan for Rs.5390 rather paying Rs.5603 after birthday. Click rml.lu/Cvs559Y here" ## spam 


# Predict the label
label, probability = model.predict(message)

print(f"Label: {label[0]}, Probability: {probability[0]}")
