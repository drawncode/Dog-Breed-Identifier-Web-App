# Dog Breed
Dog Breed is a complete and easy to use web app that can identify the breed of dogs. The code has been trained on a set of 133 different classes. Highlighting the features of web app:
  - Take a photo or browse a photo from a gallery.
  - Know the number of dogs present in the image.
  - Get a detailed information about the dogs present.

## Installations and requirements
  - Clone the repository
  ```sh
  $ git clone https://github.com/drawncode/Dog-Breed-Identifier-Web-App.git
```
  - Download the weights file of yolo using the link [YOLOv3 weights](https://pjreddie.com/media/files/yolov3.weights) or run the following command
  ```sh
  $ wget https://pjreddie.com/media/files/yolov3.weights
  ```
  - Since flask is used as a backend for the web app, make sure flask is also installed
  ```sh
  $ pip3 install Flask
```
## Instructions to run
### Using web app
  - Get your server connected to a network.
  - Note the machine's ip using the following command
  ```sh
  $ ifconfig
  ```
  - My server's ip is [10.8.18.31]()
  - Now, on the server machine, run the following command to create the server
  ```sh
  $ python3 app.py
  ```
  - Now on clients machine, open the browser.
  - Make sure no proxies are there in the browser.
  - Run the code using URL [<Your Server's ip>:50030]()
  - In my case, The URL is [10.8.18.31:50030]()
  - You are ready to use the web app.
 
### Without web app
  - Rename the test image as ***test.jpg*** and copy it to the main directory.
  - Run the command
  ```sh
  $ python3 inference.py
  ```
  - All the outputs will be saved in output directory
