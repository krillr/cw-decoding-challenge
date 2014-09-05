Codebook = {
  '.-'  :'A', '-...':'B', '-.-.':'C', '-..' :'D', '.' :'E',
  '..-.':'F', '--.' :'G', '....':'H', '..'  :'I', '.---':'J',
  '-.-':'K', '.-..' : 'L', '--' :'M', '-.' :'N', '---':'O',
  '.--.' : 'P', '--.-' : 'Q', '.-.':'R', '...':'S', '-'  :'T',
  '..-':'U', '...-' : 'V', '.--':'W', '-..-' : 'X', '-.--' : 'Y',
  '--..' : 'Z', '.----' : '1', '..---' : '2', '...--' : '3',
  '....-' : '4', '.....' : '5', '-....' : '6', '--...' : '7',
  '---..' : '8','----.' : '9','-----' : '0',
  '-...-' : '=', '.-.-':'~', '.-...' :'*', '.-.-.' : '*', '...-.-' : '*',
  '-.--.' : '*', '..-.-' : '*', '....--' : '*', '...-.' : '*',
  '.-..-.' : '\\', '.----.' : '\'', '...-..-' : '$', '-.--.' : '(', '-.--.-' : ')',
  '--..--' : ',', '-....-' : '-', '.-.-.-' : '.', '-..-.' : '/', '---...' : ':',
  '-.-.-.' : ';', '..--..' : '?', '..--.-' : '_', '.--.-.' : '@', '-.-.--' : '!', '!': ' ', '*': '*'
}

from scipy.io import wavfile
from numpy.fft import fft, fftfreq, ifft
import numpy, scipy, sys, os

MORSE_FREQUENCY = 600

def transform(d, sample_rate):
  F = fft(d)
  f = fftfreq(len(F), 1.0/sample_rate)

  return F, f

def apply_filters(transformed_signal, signal_frequencies, bandwidth=20):
  def bandpass(x,freq):
    if abs(freq)>=MORSE_FREQUENCY+bandwidth or abs(freq)<=MORSE_FREQUENCY-bandwidth:
      return 0
    else:
      return x

  F_filtered = numpy.array([bandpass(x, freq) for x, freq in zip(transformed_signal, signal_frequencies)])
  f_filtered = ifft(F_filtered)

  return f_filtered

def process(fn, bandwidth_min=5, bandwidth_max=50, bandwidth_step=5, segment_size=30, chunk_size=5):
  sample_rate, signal = wavfile.read(fn)
  chunk_size = chunk_size * sample_rate

  predictions = []

  for bandwidth in range(bandwidth_max, bandwidth_min, -1*bandwidth_step):
    prediction = ''
    remainder = ''
    for x in range(0, len(signal), chunk_size):
      chunk = signal[x:x+chunk_size]
      if not len(chunk): break

      chunk_prediction, remainder = process_chunk(chunk, sample_rate, bandwidth, segment_size, remainder)
      prediction += chunk_prediction
    if remainder and remainder in Codebook:
      prediction += Codebook[remainder]
    prediction = prediction.replace(' ','')
    predictions.append(prediction)
    if len(prediction) == 20:
      return prediction

  predictions.sort(key=lambda x:abs(20-len(x)))
  return predictions[0]

def process_chunk(chunk, sample_rate, bandwidth, segment_size, remainder):
  transformed_signal, signal_frequencies = transform(chunk, sample_rate)
  
  filtered_signal = apply_filters(transformed_signal, signal_frequencies, bandwidth)

  samples = sample_signal(filtered_signal, segment_size)
  groups = group_samples(samples)

  tones = [x[1] for x in groups if x[0]]
  pauses = [x[1] for x in groups if not x[0]]

  tonetype = numpy.mean(tones)
  charbreak = numpy.mean(pauses) + numpy.std(pauses)*0.5
  wordbreak = numpy.mean([x for x in pauses if x > charbreak])

  prediction = predict(groups, tonetype, charbreak, wordbreak, remainder)
  return prediction

def sample_signal(data, segment_size):
  samples = []

  for x in range(0, len(data), segment_size):
    samples.append(numpy.std(data[x:x+segment_size]))

  threshhold = numpy.mean(samples)
  for x in range(len(samples)):
    if samples[x] < threshhold:
      samples[x] = 0

  return samples

def group_samples(samples):
  groups = []

  length = 0
  is_tone = False
  for x in range(len(samples)):
    if (is_tone and samples[x]) or (not is_tone and not samples[x]):
      length += 1
    if length and ((is_tone and not samples[x]) or (not is_tone and samples[x])):
      if length >= 3:
        groups.append((is_tone, length))
      length = 0
    is_tone = bool(samples[x])

  return groups

def ranged_process(fn):
  for x in range(50, 10, -5):
    response = process(fn, x)
    if len(response) == 20:
      return response

def predict(groups, tonetype, charbreak, wordbreak, c=''):
  message = []
  for is_tone, length in groups:
    if is_tone:
      if length < tonetype:
        c += '.'
      else:
        c += '-'
      if len(c) == 6:
        if c in Codebook:
          message.append(c)
          c = ''
        else:
          _c = c[:]
          c = ''
          while _c:
            c += _c[-1]
            _c = _c[:-1]
            if _c in Codebook:
              message.append(_c)
              break
          if len(c) == 6:
            message.append('*')
            c = ''
    elif length >= wordbreak and (message or c):
      message.append(c)
      message.append('!')
      c = ''
    elif length >= charbreak:
      if c:
        message.append(c)
        c = ''
  s = ''.join([Codebook[x] for x in message if x in Codebook])
  return s, c

if __name__ == '__main__':
  import csv
  f = open("results.csv", 'w')
  writer = csv.writer(f)
  writer.writerow(['ID','Prediction'])
  f.flush()
  for x in range(1,201):
    if not os.path.exists("audio/cw%s.wav" % str(x).zfill(3)): continue
    response = process("audio/cw%s.wav" % str(x).zfill(3))
    writer.writerow([str(x), response])
    f.flush()
