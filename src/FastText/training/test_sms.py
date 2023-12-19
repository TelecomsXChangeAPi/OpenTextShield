import fasttext

# Load the trained model
model = fasttext.load_model('ots_sms_model_v1.bin')

# Example SMS message
# message = "Your account has been temporarily locked out now. Please log in to https://tinyurlw22222222.com/bank verify your identity." ## phishing
# message ="(9bQqme34005Uq)"
# message = "Your Microsoft verfication code is 667982"
# message = "PayPal: 441999 is your security code. Don't share your code"
# message = "Federal Credit Union ALERT: Your Credit Card has been temporarily LOCKED. Please call Services line (954) 240-5411" ## spam
# message = "Apple Notification. Your Apple iCloud ID expires today. Log in to prevent deletion http://notapple.com/user-auth/online" ## spam
# message = "No i was trying to lock down the deal yesterday :)"
message = "Want explicit SEX insecs? Ring 900123888 now! Costs 30p/min Gsex POBOX 2667 WC1N 3XX" ## spam 


# Predict the label
label, probability = model.predict(message)

print(f"Label: {label[0]}, Probability: {probability[0]}")
