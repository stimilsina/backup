##model alpha optimization for each n


import csv
import numpy as np
import matplotlib.pyplot as plt

#removes data that does not have exactly 744 time-series data of e5
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

#at each index of list, station_id,station_name is followed by 744 times series price of e5
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

#gets a list of brent prices which is constant for all the stations
def getBrent(data):
	brent=[]
	count=0
	for row in data:
		if (count==744):
			break
		brent.append(float(row[4]))
		count+=1
	brent[314]=66.995
	return brent

def addBrent(data,brent):
	list=[]
	for i in range(0,744):
		list.append(data[i])
		list.append(brent[i])
	return list


#k fold cross validation with n lagging points in ridge regression
def linear_reg(data,brent,alpha,n,fold):

	old=data

	data=addBrent(data,brent)

	len_of_fold=(len(data))/fold

	n_fold_data=[]

	y_fold_data=[]

	#dividing into n folds
	for i in range (0,fold):
		n_fold_data.append(data[(i*len_of_fold):((i+1)*len_of_fold)])
		y_fold_data.append(old[(i*(len_of_fold/2)):((i+1)*(len_of_fold/2))])

	train_mse=0
	val_mse=0
	for i in range (0,fold):
		train_x=[]
		val_x=[]
		train_y=[]
		val_y=[]
		for j in range(0,fold):
			if (i!=j):
				train_y.extend(y_fold_data[j][n:])
				for k in range(0,744/fold-n):
					train_x.append([1])
					train_x[len(train_x)-1].extend(n_fold_data[j][k*2:k*2+2*n])
			else:
				val_y.extend(y_fold_data[i][n:])
				for k in range(0,744/fold-n):
					val_x.append([1])
					val_x[k].extend(n_fold_data[j][k*2:k*2+2*n])
		train_x_t=np.transpose(train_x)
		I=np.identity(2*n+1)
		I[0][0]=0
		result=np.matmul(np.matmul(np.linalg.inv(np.matmul(train_x_t,train_x)+(alpha**2)*I),train_x_t),train_y)

		train_error=train_y-np.matmul(train_x,result)

		val_error=val_y-np.matmul(val_x,result)

		train_mse+=np.matmul(np.transpose(train_error),train_error)/(len(train_y)*fold)
		val_mse+=np.matmul(np.transpose(val_error),val_error)/(len(val_y)*fold)

	return train_mse,val_mse

#final optimal modal with alpha and n 
def final_modal(data,brent,alpha,n):
	#first n data are not used in the modal
	old=data
	data=addBrent(data,brent)
	print(len(data))
	y=old[n:]


	x=[]
	for i in range(0,744-n):
		x.append([1])
		x[i].extend(data[i*2:(i*2+2*n)])

	x_t=np.transpose(x)
	I=np.identity(2*n+1)
	I[0][0]=0
	
	result=np.matmul(np.matmul(np.linalg.inv(np.matmul(x_t,x)+(alpha**2)*I),x_t),y)

	print result

	prediction=np.matmul(x,result)

	final_mod=old[0:n]
	final_mod.extend(prediction)

	return final_mod

#returns the final best model defined by linear regression with k folds
def get_best_modal(data,brent,fold):
	n=0
	mse_train_set=[]
	mse_test_set=[]
	prev=float("inf")
	alpha=0.01
	print("STARTED!")
	for i in range(0,744/fold,3):
		mse_train_n,mse_test_n=linear_reg(data[2:],brent,alpha,i,fold)

		mse_train_alpha,mse_test_alpha=linear_reg(data[2:],brent,alpha+0.01,i,fold)

		if(mse_test_n<prev):
			n=i
			prev=mse_test_n

		if(mse_test_alpha<prev):
			alpha+=0.01
			prev=mse_test_alpha
		#print ("alpha:",0.01,"n:",n,"mse_train:",mse_train,"mse_test:",mse_test);


	result=final_modal(data[2:],brent,alpha,n)

	print("optimal n",n)
	print("optimal alpha",alpha)
	print("min error",prev)
	return result

	
data=[]
with open('csvdata.csv','rb') as csvfile:
	readCSV=csv.reader(csvfile,delimiter=',')
	count=0

	for row in readCSV:
		if (count!=0 and count<74400):
			if (row[1]!="NA" and row[2]!="NA" and row[3]!="NA" and row[4]!="NA"):
				data.append(row)
		count+=1


	brent=getBrent(data)

	data=filter_data(data)

	data=getList(data)

	#modal=get_best_modal(data[35],brent,4);#35




	modal=get_best_modal(data[29],brent,6)
	print(len(data))

	plt.figure(3).clear
	plt.ylabel('price of e5')
	plt.xlabel('timeseries')
	plt.plot(data[29][2:],label='real value')
	plt.plot(modal,label='regression estimation')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
	plt.show()

