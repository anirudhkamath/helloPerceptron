import numpy

weight = numpy.array([-1,0,0]) #weight matrix having bias weight b0=-1
x = numpy.array([[1,1,2],
				[1,-2,1],
				[1,1,0],
				[1,4,5],
				[1,2,-1],
				[1,5,1],
				[1,-1,0],
				[1,2,0],
				[1,1,1],
				[1,4,2],
				[1,1,4],
				[1,1,1],
				[1,-1,2],
				[1,-1,-1],
				[1,-1,-2],
				[1,-2,-6],
				[1,2,3],
				[1,3,3],
				[1,0.5,0.1],
				[1,0.1,0.2],
				[1,0.3,0.3],
				[1,3,4]]) #regressor, having bias input=1 ALWAYS

d = numpy.array([1,-1,-1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1]) #perceptron checks if sum>=3, 1 if true, -1 if false

def perceptron_training(weight):
	temp=0
	for i in x:
		t = numpy.dot(weight,i)
		print(t)
		y=numpy.sign(t)
		weight = weight + 0.1*(d[temp]-y)*i
		temp=temp+1
		print(weight)
	return weight

def perceptron(w):
	print("Enter 3 element query vector")
	query = [float(input()) for i in range(0,3)] #ALWAYS have bias input=1, so first element of query vector=1
	#print(query)
	for i in query:
		i = int(i)
	print(query)
	t = numpy.dot(w,query)
	print(t)
	y = numpy.sign(t)
	if y>0:
		print("Class 1, sum of numbers is >= 3")
	else:
		print("Class 0, sum of numbers is < 3")

weight1 = perceptron_training(weight)
perceptron(weight1)