import requests
response = requests.get("https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key=34b062421be395f7de0912fe00a37193&per_page=500&format=json&nojsoncallback=1&tags=forest%2C+woods&tag_mode=all&media=photos&extras=count_faves&license=7,8,9,10")

pages = response.json()["photos"]["pages"]

print(pages)

for x in range(pages):
    response = requests.get("https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key=34b062421be395f7de0912fe00a37193&per_page=500&format=json&nojsoncallback=1&tags=forest%2C+woods&tag_mode=all&media=photos&extras=count_faves&license=7,8,9,10")
    for photo in response.json()["photos"]["photo"]:
        if(int(photo["count_faves"]) >= 1):
            print(photo)
            photoResponse = requests.get("https://live.staticflickr.com/%s/%s_%s_b.jpg" % (photo["server"],photo["id"],photo["secret"]))
            open("./databaserelease2/flickr-1/%s-%s.jpg" % (photo["count_faves"],photo["id"]), 'wb').write(photoResponse.content)
