from gtts import gTTS

tts = gTTS(text='เหลืออีก30เฟรมเเล้ว',lang='th') # text คือ ข้อความ lang คือ รหัสภาษา
tts.save('FrameLeaft.mp3')