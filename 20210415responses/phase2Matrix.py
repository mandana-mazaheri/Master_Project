import os, sys
import csv
import glob
import xlsxwriter
import pandas as pd

allDatasets = []
allPipelines = []
with open(os.getcwd()+os.path.sep+'pipelines.txt') as p:
	allPipelines = p.read().splitlines()

with open(os.getcwd()+os.path.sep+'datalad_datasets.txt') as d:
	allDatasets = d.read().splitlines()





rowSize = len(allPipelines)+1
colmunSize = len(allDatasets)
pipelineDatasetArr = [[0 for i in range(len(allDatasets))] for j in range(len(allPipelines)+1)]

#[[0 for x in range(len(allPipelines+1))] for y in range(len(allDatasets))]
print (len(allPipelines)+1, len(allDatasets))
#int[len(allPipelines+1)][len(allDatasets)]
'''for i in range(len(allPipelines+1)):
	for j in range (len(allDatasets)):
		pipelineDatasetArr[i][j] = 0'''
#[ [0]*len(allPipelines+1) for _ in range(len(allDatasets)) ]





path = os.getcwd()+os.path.sep

#inputFile = glob.glob(path + 'Tristan Glatard - Evaluation.csv')
'''csvFile = open(inputFile,"r")
titlesLine = csvFile.readline().split(',')
print(titlesLine)'''
outputFile = os.getcwd()+os.path.sep+'outputFile.xlsx'
dirs = glob.glob(path +os.path.sep +'inputFiles'+os.path.sep+ '*.csv')

print(dirs)
#dirs = glob.glob(path + 'test.csv')
for file in dirs:
	inputFile = file
	csvFile = open(inputFile,"r")
	datasetsList = csvFile.readline().split(',')
	'''for d in datasetsList:
		if not d in allDatasets:
			allDatasets.append(d)'''

	#for columnIndex, dataset in enumerate(datasetsList):
		#print(str(columnIndex) +"  "+dataset)

	csv_reader = csv.reader(csvFile, delimiter=',')
	for row in csv_reader:
		pipeline = row[0]
		#print (pipeline)
		'''if not pipeline in allPipelines:
			allPipelines.append(pipeline)'''

		for index,unit in enumerate(row):
			if unit == "x" or unit == "X":
				#print(str(index) + unit)
				datasetName = datasetsList[index]
				dataset_index = allDatasets.index(datasetName.replace('\n',''))
				print(pipeline)
				print(file)
				pipeline_index = allPipelines.index(pipeline.replace('\n',''))
				pipelineDatasetArr[pipeline_index][dataset_index] += 1
				#print(datasetName + "  " + pipeline)
				#print(str(dataset_index) + "---" + str(pipeline_index))
	#print(pipelineDatasetArr)





workbook = xlsxwriter.Workbook(outputFile)
worksheet = workbook.add_worksheet()

for rowIndex in range(len(allPipelines) + 1):
	for columnIndex in range(len(allDatasets)):
		if (rowIndex == 0):
			if (columnIndex == 0):
				worksheet.write(rowIndex, columnIndex, 'Piplines \\ Datasets')
			else:
				worksheet.write(rowIndex, columnIndex, allDatasets[columnIndex - 1])
		else:
			if (columnIndex == 0):
				worksheet.write(rowIndex, columnIndex, allPipelines[rowIndex - 1])
			else:
				worksheet.write(rowIndex, columnIndex, pipelineDatasetArr[rowIndex - 1][columnIndex - 1])

workbook.close()

import os

import pandas as pd
print(os.path.join(os.getcwd(),'outputFile.csv'))
df = pd.read_csv(os.path.join(os.getcwd(),'outputFile.csv'))
df.index = df['Piplines \\ Datasets']

import plotly.express as px
fig = px.imshow(df, color_continuous_scale='Blues')
fig.show()
