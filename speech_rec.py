import speech_recognition as sr

r = sr.Recognizer()

def speech_recog_phase():
    command = None
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = r.listen(source)
        print('ok thats enough')
    try:
        command = r.recognize_whisper(audio, language="english")
        print("Whisper thinks you said " + command)
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Whisper")
    return command