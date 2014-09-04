Codebook = {
  '.-'  :'A', '-...':'B', '-.-.':'C', '-..' :'D', '.' :'E',
  '..-.':'F', '--.' :'G', '....':'H', '..'  :'I', '.---':'J',
  '-.-':'K', '.-..' : 'L', '--' :'M', '-.' :'N', '---':'O',
  '.--.' : 'P', '--.-' : 'Q', '.-.':'R', '...':'S', '-'  :'T',
  '..-':'U', '...-' : 'V', '.--':'W', '-..-' : 'X', '-.--' : 'Y',
  '--..' : 'Z', '.----' : '1', '..---' : '2', '...--' : '3',
  '....-' : '4', '.....' : '5', '-....' : '6', '--...' : '7',
  '---..' : '8','----.' : '9','-----' : '0',
  '-...-' : '=', '.-.-':'~', '.-...' :'<AS>', '.-.-.' : '<AR>', '...-.-' : '<SK>',
  '-.--.' : '<KN>', '..-.-' : '<INT>', '....--' : '<HM>', '...-.' : '<VE>',
  '.-..-.' : '\\', '.----.' : '\'', '...-..-' : '$', '-.--.' : '(', '-.--.-' : ')',
  '--..--' : ',', '-....-' : '-', '.-.-.-' : '.', '-..-.' : '/', '---...' : ':',
  '-.-.-.' : ';', '..--..' : '?', '..--.-' : '_', '.--.-.' : '@', '-.-.--' : '!', '!': ' '
}

from scipy.io import wavfile
from numpy.fft import fft, fftfreq, ifft
import numpy, sys, os

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
  p = numpy.log10(numpy.abs(f_filtered))
  power_threshhold = numpy.median(p)

  def squelch(x, power):
    if power <= power_threshhold:
      return 0
    return x

  p_filtered = numpy.array([squelch(x, power) for freq, power in zip(f_filtered, p)])

  return f_filtered

def process(fn, bandwidth_min=5, bandwidth_max=50, bandwidth_step=5, segmentSize=30):
  sample_rate, signal = wavfile.read(fn)
  transformed_signal, signal_frequencies = transform(signal, sample_rate)
  
  for bandwidth in range(bandwidth_max, bandwidth_min, -1*bandwidth_step):
    filtered_signal = apply_filters(transformed_signal, signal_frequencies, bandwidth)

    samples = sample_signal(filtered_signal, segmentSize)
    groups = group_samples(samples)

    tones = [x[1] for x in groups if x[0]]
    pauses = [x[1] for x in groups if not x[0]]

    tonetype = numpy.mean(tones)
    charbreak = numpy.mean(pauses)
    
    wordbreak = numpy.mean([x for x in pauses if x > charbreak])

    prediction = predict(groups, tonetype, charbreak, wordbreak)

    if len(prediction) == 20:
      return prediction

def sample_signal(data, segmentSize):
  samples = []

  for x in range(0, len(data), segmentSize):
    samples.append(numpy.std(data[x:x+segmentSize]))

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
      groups.append((is_tone, length))
      length = 1
    is_tone = bool(samples[x])

  return groups

def ranged_process(fn):
  for x in range(50, 10, -5):
    response = process(fn, x)
    if len(response) == 20:
      return response

def predict(groups, tonetype, charbreak, wordbreak):
  message = []
  c = ''
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
            return ''
    elif length >= wordbreak and (message or c):
      message.append(c)
      message.append('!')
      c = ''
    elif length >= charbreak:
      if c:
        message.append(c)
        c = ''
  if c: message.append(c)

  s = ''.join([Codebook[x] for x in message if x in Codebook])
  return s

if __name__ == '__main__':
  import csv
  writer = csv.writer(sys.stdout)
  writer.writerow(['ID','Prediction'])
  for x in range(1,201):
    if not os.path.exists("audio/cw%s.wav" % str(x).zfill(3)): continue
    response = process("audio/cw%s.wav" % str(x).zfill(3))
    writer.writerow([str(x), response])
