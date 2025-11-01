import os
import requests
from PIL import Image
from bs4 import BeautifulSoup

url = "https://leagueoflegends.fandom.com/wiki/Item_(League_of_Legends)"  # replace with the actual page
output_folder = "items"
os.makedirs(output_folder, exist_ok=True)

soup = BeautifulSoup(requests.get(url).text, "html.parser")
places = soup.select("div.tlist")
target = places[7]
images = target.select("ul li div div span a img")
save_path = 'items'

for image in images:
    img_location = image.get('data-src')
    img_name = image.get('data-image-name')
    img_name = img_name.replace(" item","")
    data = requests.get(img_location).content
    with open(os.path.join(save_path,img_name),'wb') as f:
        f.write(data)
