from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import os

def pdf2image(pdf_path, output_folder, poppler_path = r"C:\Users\riley\Documents\University\Honours\Honours_Project\poppler-25.12.0\Library\bin"):
# Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")
    try:
            # Convert PDF pages to a list of PIL Image objects
            # dpi=300 is standard high quality; adjust as needed
            pages = convert_from_path(pdf_path, dpi=300)

            for i, page in enumerate(pages):
                # Generate a filename for each page (e.g., page_1.png)
                image_name = f"page_{i + 1}.png"
                image_path = os.path.join(output_folder, image_name)
                
                # Save the image
                page.save(image_path, "PNG")
                print(f"Saved: {image_path}")

            print("\nConversion complete!")

    except Exception as e:
            print(f"An error occurred: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Replace with your actual file path
    target_pdf = r'C:\Users\riley\Documents\University\Honours\Honours_Project\Literature_Review\Host_galaxy_properties\betas_PLOT.pdf' 
    destination_folder = r'C:\Users\riley\Documents\University\Honours\Honours_Project\Literature_Review\Host_galaxy_properties'

pdf2image(target_pdf, destination_folder)

