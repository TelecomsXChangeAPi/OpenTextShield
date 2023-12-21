import fasttext

# Load the trained model
model = fasttext.load_model('ots_sms_model_v1.1.bin')

# Example SMS message

# Phishing Examples
# message = "URGENT: Your bank account has been compromised. To secure your funds, click here immediately: http://bit.ly/2FAKEurl"
# message = "Warning: Your email has been flagged for unusual activity. Confirm your credentials now at http://bit.ly/com32323 to avoid suspension."
message = "Paket im Terminal 2: http://different.com/7DH5000022" ## phishing

# Verification Code Examples
# message = "Google Verification Code: 452183. Do not share this code with anyone."
# message = "Your WhatsApp code: 729-113. Don't share this code with others."

# Spam Examples
# message = "Congratulations! You've won a $500 Amazon gift card. Claim now at www.win-gifts.com. Hurry, offer ends soon!"
# message = "Exclusive Offer: Get 90% off on your next purchase at SuperShoes. Visit www.someshoes.com/offer10. Unsubscribe at stopDIA.com"

# Regular Communication (Ham) Examples
# message = "Hey, are we still on for dinner tonight at 7?"
# message = "Can you send me the presentation slides? I'd like to review them before the meeting."

# More Phishing Examples
# message = "Final Notice: Your subscription with NetMedia is about to expire. Renew now to avoid service interruption: www.netmedia-renewal.com"
# message = "Your package could not be delivered due to unpaid customs fee. Pay now at www.other_domain.com/438484sjdjdjdsjjdsjdsjdjsjdsjdjsdjs"

# More Spam Examples
# message = "Hot Summers Sale! Buy one get one free on all beachwear at TrendyStyles. Shop now at www.trendystyles.com/sale"
# message = "Get a free cell data booster with every new phone plan at MobileNet. Call us at 800-555-0199 or visit www.mobilenet.com"

# More Verification Code Examples
# message = "Your Tinder verification code is 394857. Please enter this code to continue."
# message = "Dropbox: Your security code is 842159. Enter this code to complete the setup."

# More Regular Communication (Ham) Examples
# message = "Reminder: Your dentist appointment is scheduled for tomorrow at 10 AM."
# message = "Great meeting today! Let's catch up next week to discuss further steps."

# Test Data that the model have never seen before with type Phishing

# message = "ALERT: Unusual sign-in detected on your Netflix account. Verify immediately at http://netflix-verify-login.com to prevent suspension."
# message = "Your PayPal account has been temporarily restricted. Please update your information at http://paypal-secure-update.com to restore access."
# message = "Your Apple ID is due for verification. Failure to verify may lead to account suspension. Visit http://appleids-verify-n0w.com promptly."
message = "Security Alert: We've detected unusual activity on your Apple account. Please verify your information immediately at http://apple-resetpassword.com to avoid suspension."



# Predict the label
label, probability = model.predict(message)

print(f"Label: {label[0]}, Probability: {probability[0]}")
