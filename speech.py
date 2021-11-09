import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print('Say anything: ')
    audio = r.listen(source)
    try:
        text = r.recognize_bing(audio)
        print(f'You said: {text}')
    except:
        pass