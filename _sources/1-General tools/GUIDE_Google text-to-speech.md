# Google text-to-speech 
Generates audio files from text. 

The API Google Text-to-Speech (https://cloud.google.com/text-to-speech) in Google Cloud services offers advanced naturalistic voices (like Wavenet, Neural2). It also has advanced features, like the possibility of customizing and tunning voices. If you just need some basic synthetic-sounding text to speech using the standard engine, you can use'gTTS' library in Python https://pypi.org/project/gTTS/. 

In this **example** we will set up the API and run some Python code in the cloud to generate variations of a sentence for an experiment
## Get started with Google Cloud
You better follow the official quick start guides on this: https://cloud.google.com/docs/get-started
It is easy and quick to get your gmail user started. 
**Warning** note that most of the services have a cost after the free trial period. Also, of course, note that you will have your files in the Google cloud...

- Once there you can Activate 'Cloud Shell' (terminal icon on top right corner next to your user). This will open a tab at the bottom of the page where you can run commands.
- In the Cloud Shell you can also run python code if you type `iPython` 
- In the Cloud Shell terminal tab you can click *Open Editor* which you will need to run code for your API (see below) 

## Set up Google Text-to-speech API
Once you are a GC user there are many services and APIs you can use.  Text to speech is just one of them https://cloud.google.com/text-to-speech
- After becoming a Google cloud user you will probably have a default starter project "My First Project" or something like that. You will operate within that project. 

- Go to your cloud console https://console.cloud.google.com/

- In the Navigation menu on the left side go to *APIs and services/Library* 

- Once in the API library type: *Text-to-speech* in the search bar to find the API site 

- Once there follow the get started guides and click on Try This API to access the documentation: https://cloud.google.com/text-to-speech/docs/reference/rest/?apix=true

## Run code to generate sentences
- Open the **editor** (From Cloud shell terminal, click 'open editor' )
- There you should see on the left side a panel wtih your project and API folder and files 
- You can start a new file (file/ open) with some code and then run it from there  

### Code snippet 
The following code will: 
- Create variations of a fixed sentence structure in which 3 words can vary 
- Generate speech from each variation with different naturalistic voices
- Save each file as mp3 with the filename accounting for the variation number and voice used. 


````python
import google.cloud.texttospeech as tts
import os

# Sentence variations 
sentence = 'Vorsicht xnamex, gang sofort zum xcolorx Fäld vo de Spalte xnumberx'
names = ['Adler','Drossel','Tiger','Unke']
colors = ['Gelb','Gruen','Rot','Weiss']
numbers = ['Eins','Zwei','Drei','Vier']


sentence_version = [sentence.replace("xnamex", name).replace("xcolorx", color).replace("xnumberx", number)         
           for name in names for color in colors for number in numbers]

lang_voice_speaker = ['de-DE-Neural2-D',
            'de-DE-Neural2-F',
            'de-DE-Wavenet-A',
            'de-DE-Wavenet-A',
            'de-DE-Wavenet-B',
            'de-DE-Wavenet-C',
            'de-DE-Wavenet-D',
            'de-DE-Wavenet-E',
            'de-DE-Wavenet-F']
#lang_voice_speaker = ['de-DE-Wavenet-F']

def text_to_wav(voice_name: str, text: str, outputname: str):
        language_code = "-".join(voice_name.split("-")[:2])    
        text_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)
        client = tts.TextToSpeechClient()
        response = client.synthesize_speech(input=text_input, voice=voice_params, audio_config=audio_config)    
        filename = f"{outputname}.wav"
        with open(filename, "wb") as out:
            out.write(response.audio_content)
            print(f'Generated speech saved to "{filename}"')
		
for i,text in enumerate(sentence_version):
    for voice_name in lang_voice_speaker:
        outputname = "s{:02d}".format(i+1)+'_' + voice_name
        #print(outputname)
        text_to_wav(voice_name,text,outputname)

````
