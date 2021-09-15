import os
from dotenv import load_dotenv
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = "" #add between the quotes your twillio account sid
auth_token = "" #add between the quotes your twillio account token
client = Client(account_sid, auth_token)

print('Who is the arsehole without mask?!')
x = input()
print('getting this sorted now...')

if x == "fernando":
    call = client.calls.create(
                            twiml="<Response><Say> John, you do not have your mask on, please wear it now!!</Say></Response>", #change the message to be called
                            to='', #add here the number you want to call 
                            from_='' #add here your twilio number
                        )

    print(call.sid)

print('sorted ;)')
