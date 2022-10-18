# современные


# заметки
	- greedy decoder
	- beam serch decoder (итерируемся по таймстемпам)
	- beamsearch with LM
	- языковая модель выдает kenmln

	## обучение
		- Contextual Temporary Classification
		  - предсказываем для каждого фрейма свой символ
		  - расширяем словать blank символом
		  - декодируем рузультаты: удаляем подряд идущие, повторяющ символы, затем blank символы
		- аугментация: SpecCutout - зануление части спектрограммы
		- Novograd + Cosine LR (weight decay как в AdamW)
		- итоговая модель базировалась на основе QuartzNet10x5 с дополнениям


# NeMo
	
	sbertech использовали QuartzNet15x5 использует NeMo toolkit
	- https://github.com/NVIDIA/NeMo

	- Максимально просто о распознавании речи при помощи NeMo
	https://newtechaudit.ru/maksimalno-prosto-o-raspoznavanii-rechi-pri-pomoshhi-nemo/
	https://developer.nvidia.com/blog/develop-smaller-speech-recognition-models-with-nvidias-nemo-framework/
	https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html#russian

# ESPnet
	- https://github.com/espnet/espnet

# архитектура
- RNN, transformer (медленно autoregressive)
- CNN (хуже, но быстро, использовать с языковой моделью)

- первая модель на свертках (Jasper - NVIDIA 2019)
- QuartzNet (от сверток перешли к pointwise сверткам)
- ContextNet (google 2020) - глобальный контекст  
  (си блок):
    - усредняем по времени, сужаем разужаем, сигмоида
    - что-то типа аттеншна

- сверточный Jasper / его уменьшенная версия QuartzNet
- тюнят wav2vec 2.0


# пайплайн:
- препроцессинг
- акустическая модель
  - pre-emphasis filter
  - фрейминг ()
  - оконное преобразование фурье (power spectrogram) /// https://pytorch.org/audio/transforms.html
  - mel filter bank, нормализуем


# vosk
Преобразование, транскрибация и расшифровка аудио в текст с помощью Python и Vosk. Перевод русской речи в текст оффлайн.
https://proglib.io/p/reshaem-zadachu-perevoda-russkoy-rechi-v-tekst-s-pomoshchyu-python-i-biblioteki-vosk-2022-06-30


# современное
jasper 
https://arxiv.org/pdf/1904.03288.pdf
(работает с MFCC)

# Conformer Convolution-augmented Transformer for Speech Recognition
	https://arxiv.org/pdf/2005.08100.pdf


# GoodBye WaveNet -- A Language Model for Raw Audio with Context of 1/2 Million Samples
https://arxiv.org/abs/2206.08297

# wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
https://arxiv.org/pdf/2006.11477v3.pdf


# Датасеты для генерации и анализа музыки
https://neurohive.io/ru/datasety/datasety-dlya-generacii-i-analiza-muzyki/



# papers with code
https://paperswithcode.com/task/speech-recognition
	
	- wav2vec 2.0 XLS-R 1B + TEVR
	- https://paperswithcode.com/sota/speech-recognition-on-common-voice-german
	https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py


	- https://paperswithcode.com/sota/speech-recognition-on-timit
	- wav2vec 2.0 https://paperswithcode.com/paper/wav2vec-2-0-a-framework-for-self-supervised


 - какой-то фреймворк над fairseq // wav2vec
 https://github.com/facebookresearch/fairseq/
 https://github.com/mailong25/self-supervised-speech-recognition



 - quartznet15x5
 https://developer.nvidia.com/blog/develop-smaller-speech-recognition-models-with-nvidias-nemo-framework/

	# torch
	https://github.com/oleges1/quartznet-pytorch


	https://github.com/zzw922cn/Automatic_Speech_Recognition

	https://github.com/Kirili4ik/QuartzNet-ASR-pytorch

	https://github.com/qute012/korean-speech-recognition-quartznet



-  ConformerCTC-L - закрытый скрипт обучения мда
	https://github.com/sooftware/conformer

- Squeezeformer (Conformer-CTC-M с плюшками)
	https://arxiv.org/pdf/2206.00888.pdf

	# tf
	https://github.com/kssteven418/Squeezeformer
	# torch
	https://github.com/upskyy/Squeezeformer



Citrinet: Closing the Gap between Non-Autoregressive and Autoregressive End-to-End Models for Automatic Speech Recognition
https://arxiv.org/pdf/2104.01721.pdf
