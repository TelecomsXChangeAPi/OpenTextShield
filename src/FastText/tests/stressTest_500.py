import requests
import concurrent.futures
import time

# API endpoint and headers
url = 'http://localhost:8000/predict/'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}
data = '{"text":"Hello World"}'

# Function to send a single API request
def send_request():
    try:
        response = requests.post(url, headers=headers, data=data)
        return response.status_code
    except requests.exceptions.RequestException as e:
        return str(e)

# Function to send 500 requests and measure the time taken
def send_500_requests_and_measure_time():
    start_time = time.time()  # Start time

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Creating a list of 500 future objects
        futures = [executor.submit(send_request) for _ in range(500)]
        # Waiting for all the futures to complete
        concurrent.futures.wait(futures)

    end_time = time.time()  # End time

    duration = end_time - start_time
    print(f"Completed sending 500 requests in {duration:.2f} seconds")

# Send 500 requests and measure time
send_500_requests_and_measure_time()

