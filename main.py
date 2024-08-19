import cv2
import numpy as np
import fitz # PyMuPDF
import os
import uuid
import json
from PIL import Image
#from pix2tex.cli import LatexOCR
from simple_latex_ocr.models import Latex_OCR

### managing class
class TexEqManager:

    def __init__(self,pdf_path,increase_pdf_image_factor=1.0):
        self.pdf_path = pdf_path
        self.pdf_directory = os.path.dirname(pdf_path)  # Directory where the PDF is stored
        self.pdf_filename = os.path.basename(pdf_path)  # PDF file name
        self.output_folder_name = None
        self.roi_output_folder_name = None
        # json file to store various info about the pdf path
        self.json_file = os.path.join(self.pdf_directory, "config.json")  # JSON file to store the results
        # controlling image dimensions with fitz
        self.zoom_x = increase_pdf_image_factor  # Horizontal zoom
        self.zoom_y = increase_pdf_image_factor  # Vertical zoom
        self.magnifyPdfImage = fitz.Matrix(self.zoom_x, self.zoom_y)  # Create a transformation 

    def SplitPDF(self):
        print("Splitting PDF...")
         # Open the provided PDF file
        document = fitz.open(self.pdf_path)
        self.output_folder_name = f"pdf_{uuid.uuid4().hex}"
        output_folder = os.path.join(self.pdf_directory, self.output_folder_name)
         # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)
        # Iterate through each page
        for i in range(document.page_count):
            page = document.load_page(i)  # Load the current page
            pix = page.get_pixmap(matrix=self.magnifyPdfImage)  # Render page to an image pixmap

            # Define the output image path
            output_path = os.path.join(output_folder, f"page_{i + 1}.png")
            
            # Save the pixmap as a PNG
            pix.save(output_path)
            print(f"Page {i + 1} saved to {output_path}")
        # saving the name of folder for later use
        self.save_output_folder_name_to_json(self.output_folder_name, 'pdf_images')
        print("PDF split successfully")

    def SelectEquation(self):
        print("Selecting Equation...")
        if not self.output_folder_name:
            self.output_folder_name = self.load_folder_name_from_json('pdf_images')

        if not self.output_folder_name:
            print("Error: No output folder found. Please run SplitPDF first.")
            return
        
        input_folder = os.path.join(self.pdf_directory, self.output_folder_name)
        
        # Generate a new random folder name for ROI outputs
        self.roi_output_folder_name = f"roi_{uuid.uuid4().hex}"
        roi_output_folder = os.path.join(input_folder, self.roi_output_folder_name)
        
        # Create the new folder
        os.makedirs(roi_output_folder, exist_ok=True)

        # List all PNG images in the original output folder
        images = [f for f in os.listdir(input_folder) if f.endswith('.png')]
        images.sort()  # Ensure the order is correct

        # here I create the scrollable window
        #window_name = "Select ROIs"
        #self.display_scrollable(image, window_name)
        # here i plot the images and select the roi
        for image_file in images:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            #rois = self.get_rois_from_image(image, window_name)
            self.display_and_select_rois(image, image_file, roi_output_folder)

            # Save each ROI as a new image
            #for idx, roi in enumerate(rois):
            #    roi_path = os.path.join(roi_output_folder, f"{os.path.splitext(image_file)[0]}_roi_{idx + 1}.png")
            #    cv2.imwrite(roi_path, roi)

            # Optionally break the loop if needed, or continue to the next image
            #if cv2.waitKey(0) == 27:  # Press ESC to stop
            #    break
        # at the end I save the folder name to json
        self.save_output_folder_name_to_json(self.roi_output_folder_name, 'equation_folder')
        print("Equation selection completed")
    
    def display_and_select_rois(self, image, image_name, roi_output_folder):
        window_name = "Select ROIs"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 600, 600)  # Resize window to fixed size

        # Function to handle trackbar for scrolling
        scroll_offset = [0]  # Use list to allow modification in inner function
        def on_trackbar(val):
            scroll_offset[0] = val
            max_scroll = max(0, image.shape[0] - 600)
            start = min(max_scroll, val)
            end = start + 600 if start + 600 < image.shape[0] else image.shape[0]
            cropped_image = image[start:end, :]
            cv2.imshow(window_name, cropped_image)

        max_scroll = max(0, image.shape[0] - 600)
        cv2.createTrackbar('Scroll', window_name, 0, max_scroll, on_trackbar)
        on_trackbar(0)  # Initially display the image
        print("press ESC to go to the next image")
        print("press 's' to select ROI")
        roi_counter = 0  # Counter to track number of ROIs saved from this image
        while True:
            key = cv2.waitKey(0)
            if key == 27:  # ESC to quit
                break
            elif key == ord('s'):  # 's' to select ROI
                # Adjust the ROI selection according to the scroll
                roi = cv2.selectROI(window_name, image[scroll_offset[0]:scroll_offset[0]+600 if scroll_offset[0]+600 < image.shape[0] else image.shape[0], :], True, False)
                if roi != (0, 0, 0, 0):
                    x, y, w, h = roi
                    roi_counter += 1  # Increment ROI counter
                    roi_cropped = image[y + scroll_offset[0]:y + h + scroll_offset[0], x:x + w]
                    #roi_path = os.path.join(roi_output_folder, f"{os.path.splitext(image_name)[0]}_roi.png")
                    roi_filename = f"{os.path.splitext(image_name)[0]}_roi_{roi_counter}.png"
                    roi_path = os.path.join(roi_output_folder, roi_filename)
                    cv2.imwrite(roi_path, roi_cropped)
                    print(f"Saved ROI as {roi_filename}")

        cv2.destroyWindow(window_name)  # Close only the "Select ROIs" window


    def display_scrollable(self, image, window_name):
        """ Display the image with a vertical scrollbar and allow ROI selection """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 600, 600)  # Resize window to fixed size

        # Function to handle trackbar for scrolling
        def on_trackbar(val):
            max_scroll = max(0, image.shape[0] - 600)
            start = min(max_scroll, val)
            end = start + 600 if start + 600 < image.shape[0] else image.shape[0]
            cropped_image = image[start:end, :]
            cv2.imshow(window_name, cropped_image)

        # Create trackbar for scrolling
        max_scroll = max(0, image.shape[0] - 600)
        cv2.createTrackbar('Scroll', window_name, 0, max_scroll, on_trackbar)
        on_trackbar(0)  # Initially display the image


    def get_rois_from_image(self, window_name, image):
        rois = []
        # Show the image and wait for the user to select ROIs
        print("press ESC to go to the next image")
        while True:
            # selectROI window name, image, showCrosshair, fromCenter
            roi = cv2.selectROI(window_name, image, True, False)
            cv2.destroyAllWindows()  # Destroy the ROI selection window after each selection
            if roi == (0, 0, 0, 0):  # Check if ROI selection is done
                break
            # Format of roi is (x, y, w, h)
            if roi[2] and roi[3]:  # Check if width and height are not zero
                roi_cropped = image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                rois.append(roi_cropped)

        return rois
    
    def process_rois_to_latex(self):
        # here i find the image folder
        if not self.output_folder_name:
            self.output_folder_name = self.load_folder_name_from_json('pdf_images')
       
        if not self.output_folder_name:
            print("Error: No output folder found. Please run SplitPDF first.")
            return
         # here i find the roi folder
        if not self.roi_output_folder_name:
            self.roi_output_folder_name = self.load_folder_name_from_json('equation_folder')

        if not self.roi_output_folder_name:
            print("Error: No roi folder found. Please run SelectEquation.")
            return

        #if not self.roi_output_folder_name:
        #    if not self.load_configuration() or not self.roi_output_folder_name:
        #        print("Error: No ROI output folder found. Please set up or run the appropriate setup function.")
        #        return

        roi_folder = os.path.join(self.pdf_directory, self.output_folder_name, self.roi_output_folder_name)
        results_path = os.path.join(roi_folder, "latex_results.txt")

        # cheking if the roi_folder path exists
        if not os.path.exists(roi_folder):
            print(f"Error: Specified ROI folder '{roi_folder}' does not exist.")
            return

        # Initialize the model only once here
        model = Latex_OCR()

        with open(results_path, 'w') as results_file:
            for filename in os.listdir(roi_folder):
                if filename.endswith('.png'):
                    file_path = os.path.join(roi_folder, filename)
                    #img = Image.open(file_path)

                    # Process the image to get LaTeX
                    latex_result = model.predict(file_path)
                    print(f"Processing {filename}: {latex_result['formula']}, confidence: {latex_result['confidence']}")

                    # Save the filename and corresponding LaTeX code
                    results_file.write(f"{filename}: {latex_result['formula']}\n")
                else:
                    print("at the moment only png files are supported")
                    print(f"Skipping non-PNG file: {filename}")

        print(f"Saved LaTeX results to {results_path}")
        print("Processing completed")


    ## save and load from json config file
    def save_output_folder_name_to_json(self, folder_name, json_field):
        # Check if JSON file already exists
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as file:
                data = json.load(file)
        else:
            data = {}

        # Update the data with the new folder name
        data[json_field] = folder_name

        # Write the updated data back to the JSON file
        with open(self.json_file, 'w') as file:
            json.dump(data, file, indent=4)
    
    def load_folder_name_from_json(self, json_field):
        try:
            with open(self.json_file, 'r') as file:
                data = json.load(file)
                return data.get(json_field)
        except FileNotFoundError:
            print("Error: Configuration JSON file not found.")
        except json.JSONDecodeError:
            print("Error: JSON file is corrupt or empty.")


    def default_method(self):
        print("No specific method activated")



### MAIN CODE

pdf_path = "C:\\Users\\valer\\Desktop\\ML_for_robotics\\week1\\first_chapter.pdf"

# Create an instance of the class
texeq = TexEqManager(pdf_path, 4.0)

# Create an empty black image
image = np.zeros((500, 500, 3), dtype="uint8")

# Display the image
cv2.imshow("Interactive Image", image)


while True:
    # Display the image window and wait for a key press
    key = cv2.waitKey(0)

    if key == ord('1'):
        texeq.SplitPDF()
    elif key == ord('2'):
        texeq.SelectEquation()
    elif key == ord('3'):
        texeq.process_rois_to_latex()
    elif key == 27:  # ESC key
        print("Exiting...")
        break
    else:
        texeq.default_method()

    # Update the display if necessary
    cv2.imshow("Interactive Image", image)

cv2.destroyAllWindows()
