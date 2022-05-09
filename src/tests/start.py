# conda create --name al python=3.7
# conda install pytorch cudatoolkit -c pytorch
# conda install -c anaconda pandas
# conda install -c pytorch torchvision
# conda install -c conda-forge sentence-transformers
# conda install -c conda-forge matplotlib
# pip install simpletransformers
# conda install -c anaconda openpyxl

#conda create --name start python=3.7.3
#conda install -c anaconda pandas

#conda activate start
#cd TFM/acl22-revisiting-uncertainty-based-query-strategies-for-active-learning-with-transformers/MyActiveLearning/git
#python start.py
##############################imports##############################
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pdb
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn import model_selection
import torch
import pdb
import ClassificationModelExt
import classification_model_ext
##############################declaration##############################
projectName = 'MyProject'
model_name = 'all-MiniLM-L6-v2'
corpus_name = 'EU_projects'
bTest = False


##############################functions##############################
def getPath(key,tag = None, dtype = 'csv'):
	path = { 
		 'corpus': Path(f'corpus/EU_projects_dl.feather'), 
	     'keywords' : Path(f'masterdata/IA_keywords_SEAD_REV_JAG.txt'),
	     'embeddings': Path(f'corpus/corpus_embed_{ model_name }.pkl'), 
	     'histogram': Path(f'corpus/histogram.png'),
		 'datasets': Path(f'{ projectName }/{tag}/labeledDataset.{ dtype }'),
		 'model1': Path(f'{ projectName }/{tag}/mse/'),
		 'model2': Path(f'{ projectName }/{tag}/mse_weighted/')
		 
		 #'results': Path(f'data/{ projectName }/datasets/{ corpus_name }_{ value }.{ dtype }'),
		 #'plots': Path(f'data/{ projectName }/datasets/{ corpus_name }_{ value }.{ dtype }'),
		 }
	return path[key]

def inputUser(message):
	if bTest == True:
		return 
	return input(message)

def plotMeanErrorVsCosineSimilarity(df_data,predictions):
  df_data = df_data.copy()
  df_data.reset_index(inplace=True)
  str_tag = 'dl'
  losses = np.zeros(200)
  counts = np.zeros(200)
  coSimRange = np.arange(0,200)*0.01-1

  #calculate the bins
  for i,val in enumerate(coSimRange):
    df_sample = pd.DataFrame()
    df_sample = df_data[(df_data['labels']>val)&(df_data['labels']<(val+0.01))  ]
    if len(df_sample) == 0: continue

    losses[i] = np.mean(np.abs(predictions[df_sample.index] - df_sample.labels))
    counts[i] = len(df_sample)
    
    df_sample = df_sample.sample(1)

  df_Result = df_data[['labels','text']].copy()
  df_Result.insert(1, "prediction", predictions[:], True)
  df_Result.insert(2, "error", np.abs(df_Result['labels']-df_Result['prediction']), True)
  meanError = np.round(np.mean(df_Result['error']),2)
  ########################################plot data########################################


  xticks = np.array([f'{ np.round(cosSim,2) }' for cosSim in coSimRange] )
  idxS = np.where(counts!=0)[0][0]
  idxE = np.where(counts!=0)[0][-1]

  fig = plt.figure(figsize=(15,4))
  ax1 = fig.add_subplot(111)
  ax1.set_ylabel('Mean Error', color='blue')
  ax1.set_xlabel('cosine similarity of docs')
  ax1.set_xticklabels(xticks[idxS:idxE],rotation=90)
  ax1.bar(xticks[idxS:idxE], losses[idxS:idxE])
  ax2 = ax1.twinx()
  ax2.set_ylabel('Document Count', color='red')
  ax2.plot(xticks[idxS:idxE], counts[idxS:idxE], color='red',linewidth=3)
  # Add title and axis names
  #,{ titleExt }
  plt.title(f'Mean error for docs vs CosSim ({ meanError })')
  plt.show()

if __name__ == '__main__':
	##############################Output Configuration##############################
	logging.basicConfig(format='%(message)s', level=logging.INFO)
	pd.options.display.max_colwidth = 150
	pd.set_option('display.max_rows', None)
	##############################Read Corpus##############################
	inputUser(f"Let's assume we take the corpus { corpus_name } to train our models. Click Enter to presume")
	df_corpus = pd.read_feather(getPath('corpus'))
	logging.info(f'The corpus contains following data')
	df_print = df_corpus.iloc[np.r_[0:5,-5:0]][['id','text']]
	logging.info(f'{ df_print }')
	df_dataset = df_corpus.copy()
	df_dataset.drop(columns=['labels'], inplace = True)
	##############################Keword and Tag##############################
	inputUser(f"Let's assume we take the Keyword 'Deep Learning' ( Tag = 'dl' ) as label. Click Enter to presume")
	keywords = ['Deep Learning']
	str_tag = 'dl'
	##############################Cosine similarity##############################
	inputUser(f"We calculate the Cosine similarity between the Corpus and the Keword Deep Learning")
	embeddings_fname = getPath('embeddings')
	model = SentenceTransformer(model_name)
	if getPath('embeddings').exists():
		with open(embeddings_fname, "rb") as f_in:
			doc_embeddings = pickle.load(f_in)
	else:
		n_docs = len(df_dataset['text'])
		batch_size = 32
		doc_embeddings = model.encode(df_dataset['text'].values[0:n_docs],batch_size=batch_size)
		with open(embeddings_fname, 'wb') as f_Out:
			pickle.dump(doc_embeddings,f_Out)
	keyword_embeddings = model.encode(keywords)
	distances = cosine_similarity(doc_embeddings, keyword_embeddings)
	scores = np.mean(distances, axis=1)
	df_dataset.reset_index(inplace=True)
	df_dataset = pd.concat([df_dataset,pd.Series(data = scores, name = 'labels'),pd.Series(data = ((scores+1)/2), name = 'labels_zero_to_one')], axis=1)
	df_dataset_sub = df_dataset[['labels','labels_zero_to_one','text']].copy()
	df_dataset_sub.sort_values(by=['labels'],ascending=False,inplace=True)
	df_dataset_sub.reset_index(inplace=True)
	##############################Cosine similarity-part2##############################
	logging.info(f"Let's get in touch with the cosine similarity.")
	inputUser(f"What is the cosine similarity between Deep Learning and Machine Learning")
	score = cosine_similarity(keyword_embeddings, model.encode(['Machine Learning']))[0,0]
	logging.info(f"It's { np.round(score,3) }")
	inputUser(f"What is it between Deep Learning and Neural networks")
	score = cosine_similarity(keyword_embeddings, model.encode(['Neural networks']))[0,0]
	logging.info(f"It's { np.round(score,3) }")
	inputUser(f"What is it between Deep Learning and computer")
	score = cosine_similarity(keyword_embeddings, model.encode(['computer']))[0,0]
	logging.info(f"It's { np.round(score,3) }")
	inputUser(f"What is it between Deep Learning and street")
	score = cosine_similarity(keyword_embeddings, model.encode(['street']))[0,0]
	logging.info(f"It's { np.round(score,3) }")
	inputUser(f"What is it between Deep Learning and father")
	score = cosine_similarity(keyword_embeddings, model.encode(['father']))[0,0]
	logging.info(f"It's { np.round(score,3) }")
	inputUser(f"What is it between Deep Learning and beer")
	score = cosine_similarity(keyword_embeddings, model.encode(['beer']))[0,0]
	logging.info(f"It's { np.round(score,3) }")
	inputUser(f"The scale range from -1 to 1 but here all results have values between 0.15 and 0.7 Press Enter")
	getPath('datasets',str_tag).parent.mkdir(parents=True, exist_ok=True)
	#df_dataset.to_excel(getPath('datasets',str_tag), index=False)
	df_dataset.to_csv(getPath('datasets',str_tag), index=False)
	##############################check documents##############################
	idx = np.arange(0,40)
	inputUser(f"Let's check the cosine similarities of the 40 highest rated documents")
	bAnswer = 'n'
	while True:
		df_print = df_dataset_sub.iloc[idx]
		logging.info(f'{ df_print }')
		nextValHigh = np.round(np.min([df_dataset_sub.iloc[idx[0]].labels-0.05,df_dataset_sub.iloc[idx[-1]].labels]),2)
		idx = df_dataset_sub.index[df_dataset_sub.labels<nextValHigh][:40]
		message = f"Press Enter to get 40 samples from cosine similarity { nextValHigh }. Press n to get to the next step: "
		if len(idx) == 0:
			message = 'Press Enter to presume. No more samples to show.'	
		bAnswer = inputUser(message)
		if bAnswer == 'n' or len(idx) == 0:
			break
	##############################do histogram##############################
	coSimRange = np.arange(0,200)*0.01-1
	fig = plt.figure(figsize=(10,4))#figsize=(15,4)
	ax1 = fig.add_subplot(111)
	ax1.set_ylabel('Document Count', color='blue')
	ax1.set_xlabel('cosine similarity')
	ax1.hist(df_dataset['labels'],bins=coSimRange, range=(-1,1))
	ax1.plot(np.max(df_dataset['labels']), 0, marker="o", markersize=10, markeredgecolor="red")
	ax1.plot(np.min(df_dataset['labels']), 0, marker="o", markersize=10, markeredgecolor="red")
	maxV = np.round( np.max(df_dataset['labels']),2 )
	minV = np.round( np.min(df_dataset['labels']),2 )
	ax1.annotate(f'Max: { maxV }', (np.max(df_dataset['labels']), 0))
	ax1.annotate(f'Min: { minV }', (np.min(df_dataset['labels']), 0))
	plt.title(f'Distribution document count vs cosine similarity for keyword { keywords[0] }')
	plt.savefig(getPath('histogram'))
	plt.show()
	inputUser(f"Most of the documents have a similarity around 0.1. The max value is 0.62 and the lowest around -0.2. Press Enter")
	logging.info(f'Currently we are setting a threshold which determines if a document gets assigned a positive or negative weak label.')
	logging.info(f'An idea: Training the PU-learning with the soft output ( cosine similarity ) in order to enhance the training size.')
	inputUser(f"Let's call a pretrained Simpletransformer. ( It will last a few  minutes ) Press enter to resume")
	##############################Train Test##############################
	dataRatio = 1
	train_size = 0.8
	df_dataset = df_dataset[['text','labels']].copy()
	df_train, df_test = model_selection.train_test_split( df_dataset, train_size=train_size, random_state=0)
	dataCount = len(df_dataset)*dataRatio
	df_train = df_train.iloc[:int(dataCount*train_size)]
	df_test = df_test.iloc[:int(dataCount*(1-train_size))]
	##############################train Sentencetransformer##############################
	#load model and predict
	bDoesWeHaveTime = False
	if bDoesWeHaveTime:
	  model_args = ClassificationArgs()
	  model_args.num_train_epochs = 1
	  model_args.regression = True
	  model_args.overwrite_output_dir = True

	  # Create a ClassificationModel
	  model = ClassificationModelExt(
	      "roberta",
	      "roberta-base",
	      num_labels=1,
	      args=model_args
	  )

	  model.doStatistics(df_train)

	  #pdb.set_trace()
	  model.train_model(df_train)

	  # # # Evaluate the model
	  result, model_outputs, wrong_predictions = model.eval_model(df_test)
	else:
	  model = ClassificationModel(
	  	#model1
	      "roberta", getPath('model1',str_tag)
	  )
	# Make predictions with the model
	predictions_mse, raw_outputs = model.predict(list(df_test.text.values))
	plotMeanErrorVsCosineSimilarity(df_test,predictions_mse)
	inputUser(f"This network was trained for around 30 minutes. However we can see that the document count is anyhow negative correlated with the loss. Let's adjust the error function and see what happens. Press enter to resume")

	bDoesWeHaveTime = False
	if bDoesWeHaveTime:
	  model_args = ClassificationArgs()
	  model_args.num_train_epochs = 1
	  model_args.regression = True
	  model_args.overwrite_output_dir = True

	  # Create a ClassificationModel
	  model = classification_model_ext.ClassificationModelExt(
	      "roberta",
	      "roberta-base",
	      num_labels=1,
	      args=model_args
	  )

	  model.doStatistics(df_train)

	  #pdb.set_trace()
	  model.train_model(df_train)

	  # # # Evaluate the model
	  result, model_outputs, wrong_predictions = model.eval_model(df_test)
	else:
	  model = classification_model_ext.ClassificationModelExt(
	      "roberta", getPath('model2',str_tag)
	  )
	  model.doStatistics(df_train)
	# Make predictions with the model
	predictions_mse_weighted, raw_outputs = model.predict(list(df_test.text.values))
	plotMeanErrorVsCosineSimilarity(df_test,predictions_mse_weighted)
	inputUser(f"It's a little bit better. But we cannot train this models for all keyword's can we?")
