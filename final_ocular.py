import cv2
import torch
from torchvision import transforms as T
import dlib
import numpy as np
import os
import time 
from PIL import Image


# output_open_directory = "output/open"
# output_closed_directory = "output/closed"

# os.makedirs(output_open_directory, exist_ok=True)
# os.makedirs(output_closed_directory, exist_ok=True)

def final_model(input_dir,casuality):
    print("starting model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_path = "final_weights.pt"
    model = torch.jit.load(model_path)
    model = model.to(device)

    # Define the classes
    classes = ['Close-Eyes', 'Open-Eyes']

    # Define the image transformation
    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def eye_model(image):
        image = transform(image).unsqueeze(0) 
        image = image.to(device)
        model.eval()
        with torch.no_grad():  
            output = model(image)
        _, pred = torch.max(output, 1)
        return classes[pred.item()]

    # Image processing from a folder
    open_dir = 'open'
    close_dir = 'close'
    openm_count = 0
    closem_count = 0
    totalm_images = 0

    for img_name in os.listdir(input_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, img_name)

            # Load the image
            image = Image.open(img_path)

            # Get the prediction for the current image
            prediction = eye_model(image)

            # Increment counts based on the prediction
            if prediction == 'Close-Eyes':
                # image.save(f"{close_dir}/{totalm_images}.png")
                closem_count += 1
            elif prediction == 'Open-Eyes':
                # image.save(f"{open_dir}/{totalm_images}.png")
                openm_count += 1

            totalm_images += 1

    # Calculate and print the results
            
    casuality = 'human'
    if totalm_images > 0:
        print("#########################")
        print(f"Total images processed: {totalm_images}")
        # print(f"Open eyes count: {openm_count}")
        # print(f"Close eyes count: {closem_count}")
        print(f"Percentage open eyes: {openm_count / totalm_images:.2%}")
        print(f"Percentage close eyes: {closem_count / totalm_images:.2%}")
        print("#########################")

        # if casuality == 'maniken' :

        #     if float(open_count / total_images) <= 0.2 or float(close_count / total_images)  <= 0.2 :
        #         print("#### Alert ####")
        #     else :
        #         print("#### Not Alert ####") 
        # else :
        #     if float(open_count / total_images) >= 0.2 and float(close_count / total_images)  >= 0.2 :
        #         print("#### Alert ####")
        #     else :
        #         print("#### Not Alert ####") 


    else:
        print("No images found in the specified folder.")


def final_function(input_directory,casuality):
        # Path to the shape predictor model
    predictor_path = "shape_predictor_68_face_landmarks.dat" 

    # Initialize dlib's face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    open_count = 0
    close_count = 0
    total_images = 0
    all_images = 0 
    def calculate_ear(eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear

    # Threshold for deciding if eye is open or closed
    EAR_THRESHOLD = 0.21  # Below this value, the eye is considered closed

    # Eye landmark indices for the left and right eyes
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

    # Loop through all images in the directory
    for img_name in os.listdir(input_directory):
        all_images += 1
        img_path = os.path.join(input_directory, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = detector(gray)

        for face in faces:
            # Detect facial landmarks
            landmarks = predictor(gray, face)

            # Convert the facial landmarks to a numpy array
            landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Get the left and right eye coordinates
            left_eye = landmarks_points[LEFT_EYE_INDICES]
            right_eye = landmarks_points[RIGHT_EYE_INDICES]

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            # Determine which eye to process (let's use the left eye for now)
            if left_ear < EAR_THRESHOLD:
                eye_state = "closed"
            else:
                eye_state = "open"
            if right_ear < EAR_THRESHOLD:
                eye_a_state = "closed"
            else:
                eye_a_state = "open"

            # Crop the left eye region
            (x, y, w, h) = cv2.boundingRect(np.array([left_eye]))
            cropped_eye = img[y:y + h, x:x + w]

            # Save the cropped eye image in the corresponding directory
            if eye_state == "open" or eye_a_state=="open":
                open_count +=1 
                # save_path = os.path.join(output_open_directory, img_name)
            else:
                close_count +=1
                # save_path = os.path.join(output_closed_directory, img_name)
            total_images += 1
            # cv2.imwrite(save_path, cropped_eye)
            # print(f"Saved {eye_state} eye to {save_path}")
    # print((total_images/all_images)*100)
    if int((total_images/all_images)*100) > 15:
        print("#########################")
        print(f"Total images processed: {total_images}")
        # print(f"Open eyes count: {open_count}")
        # print(f"Close eyes count: {close_count}")
        print(f"Percentage open eyes: {open_count / total_images:.2%}")
        print(f"Percentage close eyes: {close_count / total_images:.2%}")
        print("#########################")
    else :
        final_model(input_dir=input_directory,casuality=casuality)

input_directory = "../my_face4" 
casuality = ''

final_function(input_directory,casuality)