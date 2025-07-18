import os
import json
import base64
import fitz  # PyMuPDF
import easyocr
from PIL import Image
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from google.cloud import vision
from google.cloud.vision_v1 import types

# Load API keys
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt():
    prompt= """
        You are a medical-bill parsing assistant who is responsible for extracting structured data from medical bills for digitisation.
you are responsible for the following tasks:
 - Determine the rotation needed (0, 90, 180, or 270 degrees clockwise) to make the image upright (portrait).
 - After ensuring portrait orientation, extract all fields exactly matching this JSON schema (no extras or omissions):

 Return the output in the following JSON format:

{
  "MetaData": {
        "invoice_no.": "",
        "invoice_date": "",
        "Sales Ex. Name": "",
        "Sales Ex. Phone No.": "",
        "SGST": "",
        "CGST": "",
        "GST %": "",
        "MSG": "",
        "TAXABLE": "",
        "SCH AMT": "",
        "TD AMT": "",
        "CD AMT": "",
        "TOTAL AMT/NET AMOUNT": "",
        "Gross": "",
        "Less": "",
        "Add": "",
        "Tot.GST": "",
        "ROUND": "",
        "Amount": ""
  
  },
  - Meta data is the invoice details such as invoice number, date, and sales executive details.Follow these rules to extract this.
  -SGST, CGST, GST %, MSG, TAXABLE, SCH AMT, TD AMT, CD AMT, TOTAL AMT/NET AMOUNT, Gross, Less, Add, Tot.GST, ROUND, Amount -These details are present below in the image or in the footer.Do not confuse.

	Amount
  Rules:
        - The metaData block can have different names so identify correct fields and fill details by finding semantics and context. 
        - The invoice number should be a unique identifier, typically a combination of letters and numbers.
        - The invoice date should be in the format "DD-MM-YYYY" or "YYYY-MM-DD" or "DD-MM-YY".
        - Add any details that is present in the invoice and if unable to  find any detail just return Unknown as value.
        - Look for invoice number near the top of the document, often prefixed with "Invoice No:", "Bill No:", or similar.
        - Date formats may vary but should be standardized to DD-MM-YYYY in the output.
        - Sales executive details are usually in a separate section or header area.
        - Sales Ex. Name is the name of sales executive and can be found in the block where invoice details are there.
        - Sales Ex. Phone No. is the phone number of sales executive and can be found in the block where invoice details are there.

  "shop":{
             "name": "",
             "address": "",
             "phone": "",
             "email": "",
             "gst_no": "",
             "DL_no": "",
             "PAN_no": "",
             "FSSAI": ""
  } ,
  -shop details is the field that contains the buyer's details such as name, address, phone, email, GST number, DL number, PAN number, and FSSAI number.Follow these rules to identify and extract this.
    Rules:
    - Buyer details contains private limited,hospital,or a name of a medical shop.
    - extract the relevant details such as name , address, phone, email, GST number, DL number, PAN number, and FSSAI number.
    - The details will be present below the shop name only.
    - Find relevant phone no. , email, GST number, DL number, PAN number, and FSSAI number from the invoice.
    - Identify shop name correctly as there are similar data related to supplier also.
    - If any of the details are not present in the invoice, just return Unknown as value.
    - Email addresses must be valid format (contain @ and domain)
    - DL_no is the Drug License number of the shop.
    - Phone numbers should be in standard format (10-12 digits)
    - GST numbers should be 15 characters.
    - PAN numbers are 10 characters.
    - FSSAI numbers -extract each detail of FSSAI number.
    - Address should include complete details including street, city, state, and PIN code.
    - Key name might be different in the invoice so identify correct fields and fill details by finding semantics and context.
    - Value is always adjacent or below the key name.
    - Always extract the whole value of the key name.
    - If value not present return Unknown.

  "supplier":{
             "name": "",
             "address": "",
             "phone": "",
             "email": "",
             "gst_no": "",
             "DL_no": "",
             "PAN_no": "",
             "FSSAI": ""
             }
    -supplier details is the field that contains the details of distributor such as name, address, phone, email, GST number, DL number, PAN number, and FSSAI number.Follow these rules to identify and extract this.
    Rules:
    - Supplier contains words such as distributor,llp etc.
    - extract the relevant details such as name , address, phone, email, GST number, DL number, PAN number, and FSSAI number.
    - Extract relevant details that is not mentioned in the above json if necessary.
    - Apply the same validation rules as shop details for email, phone, GST, DL, PAN, and FSSAI numbers.
    - Supplier details are usually in the header or footer of the document
    - If value not present return Unknown.

 "medecines":[
             {"product_name": "",
              "batch_no": "",
              "product_power": "",
              "PACK":"",
              "MFG":"",
              "TD":"",
              "CD":"",
              "Taxable":"",
              "HSN":"",
              "Discount":"",
              "LOC":"",
              "expiry_date": "",
              "quantity": "",
              "MRP": "",
              "Rate": "",
              "GST %":"",
              "GST": "",
              "CGST %": "",
              "SGST %": "",
              "CGST": "",
              "SGST": "",
              "amount": ""
              },
              {"product_name": "",
              "batch_no": "",
              "product_power": "",
              "PACK":"",
              "MFG":"",
              "TD":"",
              "CD":"",
              "Taxable":"",
              "HSN":"",
              "Discount":"",
              "LOC":"",
              "expiry_date": "",
              "quantity": "",
              "MRP": "",
              "Rate": "",
              "GST %":"",
              "GST": "",
              "CGST %": "",
              "SGST %": "",
              "CGST": "",
              "SGST": "",
              "amount": ""
              },
              ...
        For medecines the details are mentioned in the table format in the invoice.Follow these rules to identify and extract this.
    Rules:
    - Extract all the details of the medicines from the invoice.
    - Do not misspell the name of the medicine.
    - Use the correct column values for:

        - "GST" → only value from column labeled GST.Can be named as GSTAMT etc.
        -"GST %"- It is the value present below "%" sign beside gst.
        - "CGST %"-It is the value present below "%" sign beside cgst.
        - "CGST" → only from CGST
        - "SGST %"-It is the value present below "%" sign beside sgst.
        - "SGST" → only from SGST
        Do not mix or sum them.
    - Name of the medecine will be under the column with name product or medicine name or Description.Name might differ from invoice to invoice,so find it.
    - Extract "product_power" directly from the name of the medecine only. If not available, return "Unknown".The unit will be mg only and nothing else.
    - The relevant details of the medecines shall be corresponding to each other.
    - extract all medecine details in form of array inside medecines key.
    - If any of the details are not present in the invoice, just return Unknown as value.
    - Ensure correct naming of the fields as per the schema.
              ]
}

Return a single JSON object in this format:
{{
  "data": {{ /* extracted fields */ }}
}}    
    """
    return prompt
# Step 1: PDF to Image

def pdf_to_image(pdf_path, image_path):
    doc = fitz.open(pdf_path)
    pix = doc.load_page(0).get_pixmap(dpi=300)
    pix.save(image_path)
    print("[✔] Stage 1: Saved raw image as:", image_path)

# Step 2: Auto-Rotate Using Google Vision

def autorotate_with_google(image_path, output_path):
    vision_client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as img_file:
        content = img_file.read()
    image = types.Image(content=content)
    response = vision_client.document_text_detection(image=image)

    angle = 0
    try:
        vertices = response.text_annotations[0].bounding_poly.vertices
        x0, y0 = vertices[0].x, vertices[0].y
        x1, y1 = vertices[1].x, vertices[1].y
        dx = x1 - x0
        dy = y1 - y0

        if abs(dx) > abs(dy):
            if dx > 0 and dy < 0:
                angle = 270
            elif dx < 0 and dy > 0:
                angle = 90
        elif dy < 0:
            angle = 180
        else:
            angle = 0
    except Exception as e:
        print("Rotation detection failed, defaulting to 0°.", e)
        angle = 0

    img = Image.open(image_path)
    if angle != 0:
        rotated = img.rotate(-angle, expand=True)
        rotated.save(output_path)
        print(f" Rotated {angle}° to portrait and saved as: {output_path}")
    else:
        img.save(output_path)
        print(" No rotation needed. Saved original as:", output_path)

    return angle, output_path

# Step 3: Grayscale + Contrast Enhancement

def convert_to_grayscale(image_path, output_path):
    img = Image.open(image_path).convert("L")
    enhanced = img.point(lambda x: 0 if x < 140 else 255, mode='1')
    enhanced.save(output_path)
    print("Stage 3: Saved contrast-enhanced grayscale image as:", output_path)
    return output_path

# Step 3.5: Zoom the Image

def zoom_image(image_path, output_path, scale_factor=1.5):
    img = Image.open(image_path)
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    zoomed = img.resize(new_size, Image.LANCZOS)
    zoomed.save(output_path)
    print(f"Zoomed image saved at {output_path}")
    return output_path

# Step 4: EasyOCR

def extract_text_easyocr(image_path, text_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path, detail=0, paragraph=True)
    text = "\n".join(result)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Stage 4: Saved extracted text as:", text_path)
    return text

# Step 5: GPT-4.1 Vision

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def call_gpt_vision(base64_image, extracted_text, rotation_angle):
    prompt = build_prompt()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"The OCR extracted text is:\n{extracted_text}\n\nRotation: {rotation_angle} degrees."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=2048
    )
    print("Stage 5: GPT-4 response received.")
    return response.choices[0].message.content

# Master Pipeline

def process_pipeline(pdf_path):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    raw_image_path = os.path.join(output_dir, f"{filename}_stage1_raw.jpg")
    rotated_image_path = os.path.join(output_dir, f"{filename}_stage2_rotated.jpg")
    grayscale_image_path = os.path.join(output_dir, f"{filename}_stage3_grayscale.jpg")
    zoomed_image_path = os.path.join(output_dir, f"{filename}_stage4_zoomed.jpg")
    ocr_text_path = os.path.join(output_dir, f"{filename}_stage5_text.txt")
    output_json_path = os.path.join(output_dir, f"{filename}.json")

    pdf_to_image(pdf_path, raw_image_path)
    rotation_angle, rotated_image = autorotate_with_google(raw_image_path, rotated_image_path)
    grayscale_image = convert_to_grayscale(rotated_image, grayscale_image_path)
    zoomed_image = zoom_image(grayscale_image, zoomed_image_path)
    extracted_text = extract_text_easyocr(zoomed_image, ocr_text_path)
    base64_image = image_to_base64(zoomed_image)
    gpt_response = call_gpt_vision(base64_image, extracted_text, rotation_angle)

    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(gpt_response)
    print(f"Final Output saved to {output_json_path}")

if __name__ == "__main__":
    process_pipeline("supplierbillformatsforocr/WhatsApp Image 2025-06-12 at 12.55.29_8f126d18.jpg")
    
