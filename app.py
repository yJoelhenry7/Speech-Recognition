from flask import Flask, render_template,request,redirect
from werkzeug.utils import secure_filename
import pickle
import librosa
from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
import soundfile
import noisereduce
with open('./model.pkl',"wb") as f:
   pickle.dump("model",f)

app = Flask(__name__,static_folder='staticFiles')

# Routes
@app.route('/',methods = ['GET','POST'])
def index():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file1():
   if request.method == 'POST':
      # If file is not selected this will take back them to homepage
      if "file" not in request.files:
          request.redirect(request.url)
      file = request.files["file"]
      if file.filename == "":
         return redirect(request.url) 
      file.save(secure_filename(file.filename))
      print(file)
      loaded_model = pickle.load(open('model.pkl', 'rb'))
      return render_template('result.html')
   
# def extract_feature(file_name, mfcc, chroma, mel):
#     with soundfile.SoundFile(file_name) as sound_file:
#         X = sound_file.read(dtype="float32")
#         sample_rate=sound_file.samplerate
#         if chroma:
#             stft=np.abs(librosa.stft(X))
#         result=np.array([])
#         if mfcc:
#             mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=40).T, axis=0)
#             result=np.hstack((result, mfccs))
#         if chroma:
#             chroma=np.mean(librosa.feature.chroma_stft(S=stft,sr=sample_rate).T,axis=0)
#             result=np.hstack((result, chroma))
#         if mel:
#             mel=np.mean(librosa.feature.melspectrogram(X,sr=sample_rate).T,axis=0)
#             result=np.hstack((result, mel))
#     return result
   
# def load_data(filep):
#   x,y=[],[]
#   sound = AudioSegment.from_wav(filep)
#   sound = sound.set_channels(1)
#   sound.export(filep, format="wav")
#   rate, data = wavfile.read(filep)
#   reduced_noise = noisereduce.reduce_noise(y=data, sr=rate)
#   feature = extract_feature(filep, mfcc=True, chroma=True, mel=True)
#   x.append(feature)
#     #y.append(emotion)
#   return np.array(x)

if __name__ == '__main__':
    app.debug = True
    app.run()