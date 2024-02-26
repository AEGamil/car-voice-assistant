from TW_detection import trigger_word_phase
from speech_rec import speech_recog_phase
from intention_recog import intention_recognition
import time

intention_recog_phase = intention_recognition()

while True:
    if trigger_word_phase():
        command = speech_recog_phase()
        intention_recog_phase.go(command)
    time.sleep(4)
