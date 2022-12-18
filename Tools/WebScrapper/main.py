import requests
from bs4 import BeautifulSoup
import os
import pickle
import requests
import cv2

# Set the search term and number of images to download
search_term = "silueta de moto deportiva"
# Set the URL for the search
url = f"https://www.google.com/search?q={search_term}&source=lnms&tbm=isch"
# Send the request to the URL and get the HTML response
response = requests.get(url)
# Parse the HTML response
soup = BeautifulSoup(response.text, "html.parser")
# Find all the image tags
images = soup.find_all("img")
# Extract the image URLs
image_urls = []
for image in images:
    src = image.get("src")
    if src and ".gif" not in src:
        image_urls.append(src)

# Create a directory to hold the data
if not os.path.exists("data"):
    os.makedirs("data")

# Download and save the images
for i, image_url in enumerate(image_urls):
    response = requests.get(image_url)
    if not os.path.exists(f"data/{search_term}"):
         os.makedirs(f"data/{search_term}")
    with open(f"data/{search_term}/{search_term}-{i}.jpg", "wb") as f:
        f.write(response.content)
        f.close()
        # Load the image using cv2
        image = cv2.imread(f"data/{search_term}/{search_term}-{i}.jpg")
        # Resize the image to 32x32 pixels
        image = cv2.resize(image, (32, 32))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Save the resized image to a new file
        cv2.imwrite(f"data/{search_term}/{search_term}-{i}.jpg", image)

    

# Serialize the data and save it to a file
data = {"image_urls": image_urls}
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)
