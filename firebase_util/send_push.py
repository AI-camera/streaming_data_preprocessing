import firebase_admin
from firebase_admin import credentials, messaging

cred = credentials.Certificate("./firebase_util/serviceAccountKey.json")

firebase_admin.initialize_app(cred)
token = ["fd-pjAU6NoVoWzbYNJkBCP:APA91bFgYcOoMP_Nu0NjWViZTwVVg8o7JnaVWTncxtcV9v8fGqZsE4Ao8j7rhxO9fcJoImpCUf4bufl2A3OuzSi91VreclPAFVsTLw_49kp9BBMEaoUp7AaiZLZk4AmG3ZN-D_ZvRH8z"]

def sendPush(title, msg, registration_token=token, dataObject=None):
    # See documentation on defining a message payload.
    message = messaging.MulticastMessage(
        notification=messaging.Notification(
            title=title,
            body=msg,
            # image='./template/hlv.jpg'
            # image='https://cdn-img.thethao247.vn/storage/files/viettq/2021/10/17/hlv-park-hang-seo-2-1634457735.jpg'
        ),
        data={
            "title": title,
            "body": msg,
            "src":"edge_device"
            # "image": './template/hlv.jpg'
            # "image":'https://cdn-img.thethao247.vn/storage/files/viettq/2021/10/17/hlv-park-hang-seo-2-1634457735.jpg'
        },
        tokens=registration_token,
    )

    # Send a message to the device corresponding to the provided
    # registration token.
    response = messaging.send_multicast(message)
    if response.failure_count > 0:
        responses = response.responses
        failed_tokens = []
        for idx, resp in enumerate(responses):
            if not resp.success:
                failed_tokens.append(registration_token[idx])
        print('List of tokens that caused failures: {0}'.format(failed_tokens))
    else:
        # Response is a message ID string.
        print('Successfully sent message:', response)

if __name__ == "__main__":
    # token = ["fd-pjAU6NoVoWzbYNJkBCP:APA91bFgYcOoMP_Nu0NjWViZTwVVg8o7JnaVWTncxtcV9v8fGqZsE4Ao8j7rhxO9fcJoImpCUf4bufl2A3OuzSi91VreclPAFVsTLw_49kp9BBMEaoUp7AaiZLZk4AmG3ZN-D_ZvRH8z"]
    token = ["c3mldnxdTCuj7zLF8DJ-f_:APA91bFgiQ9D87qOB73sxuLz0Ot1qO2ToDhXz4Z2jc50ZvDDGs5E8FTNqBNMvh1BDRvoJQIWNeOQ3FwTyHxyaGYGYoTHrG7gVvSMOjSziESKfKdxtG71-Szv7n8ZDFodkwoABv9XNdri"]
    sendPush("pi", "cháo lòng không lòng",token)