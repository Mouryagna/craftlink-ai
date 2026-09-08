from pathlib import Path
from gtts import gTTS

BASE_DIR = Path.cwd()
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_OUTPUT_PATH = MEDIA_DIR / "artisan_sample.mp3"

# Realistic artisan voice script (Hindi) with craft process, time, materials, and dimensions
ARTISAN_SCRIPT = (
    "नमस्ते। मेरा नाम रामू है। मैंने यह हस्तनिर्मित टेराकोटा मिट्टी का फूलदान बनाया है। "
    "इसको बनाने में मुझे पूरे चार दिन का समय लगा है। "
    "इसमें हमने शुद्ध नदी किनारे की चिकनी मिट्टी और प्राकृतिक लाख के रंगों का उपयोग किया है। "
    "चाक पर आकार देने के बाद इसे पारंपरिक भट्टी में पकाया गया और फिर हाथों से बारीक नक्काशी की गई। "
    "इसकी ऊंचाई लगभग पच्चीस सेंटीमीटर है और चौड़ाई चौदह सेंटीमीटर है। "
    "यह पूरी तरह से पर्यावरण अनुकूल और टिकाऊ है।"
)

def create_sample_audio(output_path: Path = AUDIO_OUTPUT_PATH):
    print("[-] Generating realistic Hindi artisan voice audio...")
    tts = gTTS(text=ARTISAN_SCRIPT, lang="hi", slow=False)
    tts.save(str(output_path))
    print(f"[+] Audio successfully saved to: {output_path}")

if __name__ == "__main__":
    create_sample_audio()