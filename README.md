# Virtual Try On

Welcome to the Virtual Try On project! This application allows you to virtually try on various items using your webcam. Below are the instructions to set up and run the project.

## Technologies Used
- HTML
- CSS
- JavaScript
- Python
- Flask


## Dependencies
Make sure you have the following dependencies installed before running the application:

- python
- OpenCV (CV2)
- Mediapipe
- Flask

# How to Run 
You can install them using the following commands:
```bash
pip install opencv-python
pip install mediapipe
pip install Flask
```

Clone the repository to your local machine:
```bash 
git clone https://github.com/RonakR68/26_Virtual-Try-on
cd 26_Virtual-Try-on
```

run the flask server by execting the following command:
```python
python3 app.py
```

open your web brower and navigate to the following address :
http://127.0.0.1:5000

Explore the user-friendly fronted and enjoy the Virtual Try On website 

### Note 
- Make sure that no other application is using the camera at the moment 
- if you want to use an external camera , you need to make changes in the `cv.VideoCapture(0)` line in `app.py` file 
