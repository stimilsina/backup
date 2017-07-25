import csv
import numpy as np
import matplotlib.pyplot as plt

def filter_data(data):
	res=[]
	valid=[]
	prev=data[0][1]
	for row in data:
		if (prev==row[1]):
			valid.append(row)
		else:
			if (len(valid)==744):
				res.append(valid)
			valid=[]
			prev=row[1]
			valid.append(row)
	return res

def getList(data):
	res=[]
	station=[]
	count=0
	for row in data:
		station.append (row[0][1])
		station.append (row[0][2])
		for x in row:
			station.append(float(x[3]))
		res.append(station)
		station=[]
		count+=1
	return res

def getBrent(data):
	brent=[]
	count=0
	for row in data:
		if (count==744):
			break
		brent.append(float(row[4]))
		count+=1
	return brent


def linear_reg(data,alpha,n,fold):

	len_of_fold=len(data)/fold

	n_fold_data=[]

	#dividing into folds
	for i in range (0,fold):
		n_fold_data.append(data[(i*len_of_fold):((i+1)*len_of_fold)])

	print len(n_fold_data[0])
	print len(n_fold_data[1])

	#first data y=result
	y=n_fold_data[0][n:]

	x=[]
	for i in range(n,len_of_fold):
		x.append([1])

	for i in range(n,len_of_fold):
		x[i-n].extend(n_fold_data[0][i-n:i])

	x_t=np.transpose(x)
	I=np.identity(n+1)
	result=np.matmul(np.matmul(np.linalg.inv(np.matmul(x_t,x)+(alpha**2)*I),x_t),y)

	error=y-np.matmul(x,result)

	mse=np.matmul(np.transpose(error),error)

	mse= mse/(len_of_fold-n)

	x_val=[]
	for i in range(n,len_of_fold):
		x_val.append([1])
		x_val[i-n].extend(n_fold_data[1][i-n:i])
	error_val=n_fold_data[1][n:]-np.matmul(x_val,result)
	mse_val=np.matmul(np.transpose(error_val),error_val)
	avg_mse_val=(mse_val/(len_of_fold-n)+mse)/2

	x.extend(x_val)
	return mse, np.matmul(x,result)

data=[]
with open('csvdata.csv','rb') as csvfile:
	readCSV=csv.reader(csvfile,delimiter=',')
	count=0

	for row in readCSV:
		if (count!=0 and count<1000000):
			if (row[1]!="NA" and row[2]!="NA" and row[3]!="NA" and row[4]!="NA"):
				data.append(row)
		count+=1

	brent=getBrent(data)

	data=filter_data(data)

	data=getList(data)

	j=0.1

	alpha=np.linspace(0.1,10,100)
	n=np.arange(1,101)
	print n
	
	min_e=10000
	opt_alpha=-1
	opt_n=-1
	result=[]
	print("min error= ",min_e)
	print(" alpha= ",opt_alpha)
	print(" opt_n= ",opt_n)
	'''
	for i in n:
		for j in alpha:
			mse,res=linear_reg(data[10][2:],j,i,2)
			if (mse<min_e):
				min_e=mse
				opt_alpha=j
				opt_n=i
				result=res
				print("min error= ",min_e)
				print(" alpha= ",opt_alpha)
				print(" opt_n= ",opt_n)
	print("min error= ",min_e)
	print(" alpha= ",opt_alpha)
	print(" opt_n= ",opt_n)
	'''
	mse,result=linear_reg(data[10][2:],0,1,2)
	'''
	for i in range(5,100,10):
		m,r=linear_reg(data[100][2:],0.05,5,2)
		print (m)
		if (m>=mse):
			break
		mse,result=m,r
	'''
	print(len(data))
	plt.plot(data[10][2:])
	plt.plot(result)
	plt.show()
	'''
	print(data[100][1])
	
	plt.figure(1).clear()

	plt.plot(data[100][2:])

	print(data[10][1])

	plt.figure(2).clear
	plt.plot(data[10][2:])
	print(data[200][1])

	plt.figure(3).clear
	plt.plot(data[200][2:])
	plt.show()
	
	fit=np.polyfit(brent,data[100][2:],1)
	fit_fn=np.poly1d(fit)
	plt.plot(brent,data[200][2:],brent,fit_fn(brent))
	plt.xlim(min(brent),max(brent))
	plt.show()
	'''
    
