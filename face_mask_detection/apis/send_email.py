import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import base64 as bs
from sendgrid.helpers.mail import (Mail, Attachment, FileContent, FileName, FileType, Disposition)
from fastapi import FastAPI
from fastapi import FastAPI, Body, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8005/video.html",
    "http://localhost:8005",
    "https://localhost",
    "https://localhost:8000",
    "https://localhost:8005",
    #actual url in production,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/files")
async def create_file(blob: UploadFile = File(...)):
    #import ipdb; ipdb.set_trace()
    #print(blob)
    content = await blob.read() 
    bytes_stream = io.BytesIO(content)
    print(bytes_stream)
    img = Image.open(bytes_stream)
    print(img)
    img.save('hey.png')
    sending_email("hey.png", "fernandonjardim@gmail.com")
    return {"status": "image sent"}

#@app.post("/uploadfile/")
#async def create_upload_file(file: UploadFile = File(...)):
#   return {"filename": file.filename}

def sending_email(image, email_address):
    message = Mail(
        from_email='emaildetendencias@gmail.com',
        to_emails= email_address,
        subject='Mask Alert',
        html_content='<strong>Someone at the living room has no mask</strong>')

    with open(image, 'rb') as f:
         data = f.read()
         f.close()
         encoded_file = bs.b64encode(data).decode()
         print("encoded_file", encoded_file)

    attachedFile = Attachment(
        FileContent(encoded_file),
        FileName('1.jpg'),
        FileType('application/jpg'),
        Disposition('attachment')
    )
    message.attachment = attachedFile

    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)

"""
def sending_email():
    message = Mail(
        from_email='emaildetendencias@gmail.com',
        to_emails= "fernandonjardim@gmail.com",
        subject='Sending with Twilio SendGrid is Fun',
        html_content='<strong>and easy to do anywhere, even with Python</strong>')
    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)

sending_email()
"""
