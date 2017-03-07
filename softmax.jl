### 
###  Please refer http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/ for more detailed explanation
###

for p in ("Compat","GZip")
	Pkg.installed(p) == nothing && Pkg.add(p)
end

using Compat, GZip

function loaddata()
	info("Loading MNIST...")
	xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
	xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
	ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
	ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]
	return (xtrn, ytrn, xtst, ytst)
end

function gzload(file; path="$file", url="http://yann.lecun.com/exdb/mnist/$file")
	isfile(path) || download(url, path)
	f = gzopen(path)
	a = @compat read(f)
	close(f)
	return(a)
end

function minibatch(X, Y, bs=100)
	#takes raw input (X) and gold labels (Y)
	#returns list of minibatches (x, y)
	data = Any[]
	
	#start of step 1
	# YOUR CODE HERE
	
	X = reshape(X, size(X,1), Int(size(X,2)/bs), bs)
	println("x size", size(X))
	Y = reshape(Y, size(Y,1), Int(size(Y,2)/bs), bs)
	println("y size", size(Y))
	
	for i=1:bs
		pairs = [X[:,:,i],Y[:,:,i]]
		push!(data,pairs)
	end

	#end of step 1
	println("minibatches", size(data))
	
	return data
end

function init_params(ninputs, noutputs)
	#takes number of inputs and number of outputs(number of classes)
	#returns randomly generated W and b(must be zeros vector) params of softmax
	
	#start of step 2
	# YOUR CODE HERE
	w = 0.1*randn(noutputs, ninputs)
	b = zeros(noutputs)
	
	return w,b
	#end of step 2
end

function softmax_cost(W, b, data, labels)
	#takes W, b paremeters, data and correct labels
	#calculates the soft loss, gradient of w and gradient of b

	#start of step 3
	# YOUR CODE HERE
	
	#soft loss
	soft_probs = softmax_forw(W, b, data)
	nll = -1*sum(labels.*log(soft_probs))/ size(labels,2) #one hot labels
	
	#println("softmax_cost, labels size: ",size(labels))
	#println("softmax_cost, label-probs size: ",size(soft_probs))
	g_y = (labels - soft_probs) / size(labels,2) #gradiend of J w/r to y with normalized labels
	g_b = sum(g_y,2)
	g_w = g_y * data'
	
	#println("softmax_cost, gy size: ",size(g_y))
	#println("softmax_cost, data size: ",size(data))
	return nll, g_w, g_b
	#end of step 3
end

function softmax_forw(W, b, data)
	#applies affine transformation and softmax function
	#returns predicted probabilities
	
	### step 3
	# YOUR CODE HERE
	#println("softmax data size: ", size(data))
	y = W*data .+ b
	expy = exp(y)
	#println("ab: ", size(expy))
	softprobs = expy./sum(expy,1)
	
	#println("softmax, label-probs size: ",size(softprobs))
	return softprobs
	### step 3
end

function accuracy(ygold, yhat)
	correct = 0.0
	for i=1:size(ygold, 2)
		correct += indmax(ygold[:,i]) == indmax(yhat[:, i]) ? 1.0 : 0.0
	end
	return correct / size(ygold, 2)
end

function grad_check(W, b, data, labels)
	function numeric_gradient()
		epsilon = 0.0001

		gw = zeros(size(W))
		gb = zeros(size(b))
		
		#start of step 4
		# YOUR CODE HERE
		for j = 1:size(gw,2)
			for i = 1:size(gw,1)
				delta_wij = W
				delta_wij[i,j] = W[i,j] .+ epsilon
				J_pos,_,_ = softmax_cost(delta_wij , b, data, labels)
				delta_wij[i,j] = W[i,j] .- epsilon
				J_neg,_,_ = softmax_cost(delta_wij , b, data, labels)
				
				gw[i,j] += (J_pos - J_neg)/(2*epsilon)
			end
		end
		
		for i = 1:size(gb,1)
			delta_b = b
			delta_b[i] = b[i] .+ epsilon
			J_pos,_,_ = softmax_cost(W , delta_b, data, labels)
			delta_b[i] = b[i] .- epsilon
			J_neg,_,_ = softmax_cost(W , delta_b, data, labels)
			
			
			gb += (J_pos - J_neg)/(2*epsilon)
		end
		#end of step 4

		return gw, gb
	end

	_,gradW,gradB = softmax_cost(W, b, data, labels)
	gw, gb = numeric_gradient()
	
	diff = sqrt(sum((gradW - gw) .^ 2) + sum((gradB - gb) .^ 2))
	println("Diff: $diff")
	if diff < 1e-7
		println("Gradient Checking Passed")
	else
		println("Diff must be < 1e-7")
	end

end

function train(W, b, data, lr=0.15)
	totalcost = 0.0
	numins = 0
	for (x, y) in data
		c, gw, gb = softmax_cost(W, b, x, y)
		totalcost += c
		#println(size(W))
		#println(size(gw))
		copy!(W, W + lr * gw)			#changed this part pf the original code from - to +
		copy!(b, b + lr * gb)

		totalcost += c * size(x, 2)
		numins += size(x, 2)
	end
	avgcost = totalcost / numins
end

function main()
	srand(12345)
	# Size of input vector (MNIST images are 28x28)
	ninputs = 28 * 28

	# Number of classes (MNIST images fall into 10 classes)
	noutputs = 10

	## Data loading & preprocessing
	#
	#  In this section, we load the input and output data, 
	#  prepare data to feed into softmax model.
	#  For softmax regression on MNIST pixels,
	#  the input data is the images, and
	#  the output data is the labels.
	#  Size of xtrn: (28,28,1,60000)
	#  Size of xtrn must be: (784, 60000)
	#  Size of xtst must be: (784, 10000)

	xtrnraw, ytrnraw, xtstraw, ytstraw = loaddata()
	
	xtrn = convert(Array{Float32}, reshape(xtrnraw ./ 255, 28*28, div(length(xtrnraw), 784)))
	ytrnraw[ytrnraw.==0]=10;
	ytrn = convert(Array{Float32}, sparse(convert(Vector{Int},ytrnraw),1:length(ytrnraw),one(eltype(ytrnraw)),10,length(ytrnraw)))
	
	xtst = convert(Array{Float32}, reshape(xtstraw ./ 255, 28*28, div(length(xtstraw), 784)))
	ytstraw[ytstraw.==0]=10;
	ytst = convert(Array{Float32}, sparse(convert(Vector{Int},ytstraw),1:length(ytstraw),one(eltype(ytstraw)),10,length(ytstraw)))

	## STEP 1: Create minibatches
	# Complete the minibatch function
	# It takes the input matrix (X) and gold labels (Y)
	# returns list of tuples contain minibatched input and labels (x, y)
	bs = 100
	trn_data = minibatch(xtrn, ytrn, bs)

	## STEP 2: Initialize parameters
	#  Complete init_params function
	#  It takes number of inputs and number of outputs(number of classes)
	#  It returns randomly generated W matrix and bias vector

	W, b = init_params(ninputs, noutputs)
	
	## STEP 3: Implement softmax_forw and softmax_cost
	#  softmax_forw function takes W, b, and data
	#  calculates predicted probabilities
	#  
	#  softmax_cost function obtains probabilites by calling softmax_forw
	#  then calculates soft loss and
	#  gradient of W and gradient of b

	## Gradient checking
	#  Skip this part for the lab session, but complete later.
	#  As with any learning algorithm, you should always check that your
	#  gradients are correct before learning the parameters.

	debug = true #Turn this parameter off, after gradient checking passed
	
	if debug
		grad_check(W, b, xtrn[:, 1:100], ytrn[:, 1:100])
	end
	
	lr = 0.15

	## Training
	#  The train function takes model parameters and the data
	#  Trains the model over minibatches
	#  For each minibatch, cost and gradients are calculated then model parameters updated
	#  train function returns average cost
	
	for i=1:50
		cost = train(W, b, trn_data, lr)
		pred = softmax_forw(W, b, xtrn)
		trnacc = accuracy(ytrn, pred)
		pred = softmax_forw(W, b, xtst)
		tstacc = accuracy(ytst, pred)
		@printf("epoch: %d softloss: %g trn accuracy: %g tst accuracy: %g\n", i, cost, trnacc, tstacc)
	end

end

main()

#Example experiment log:
#===========================
Diff: 5.896954939041637e-10
Gradient Checking Passed
epoch: 4 softloss: 0.645301 trn accuracy: 0.86665 tst accuracy: 0.8734
epoch: 5 softloss: 0.592848 trn accuracy: 0.873033 tst accuracy: 0.8794
epoch: 6 softloss: 0.554782 trn accuracy: 0.877483 tst accuracy: 0.8829
epoch: 7 softloss: 0.525297 trn accuracy: 0.8818 tst accuracy: 0.8859
epoch: 8 softloss: 0.501499 trn accuracy: 0.885233 tst accuracy: 0.8886
epoch: 9 softloss: 0.481751 trn accuracy: 0.888167 tst accuracy: 0.8904
epoch: 10 softloss: 0.465025 trn accuracy: 0.89095 tst accuracy: 0.8918
epoch: 11 softloss: 0.450627 trn accuracy: 0.892883 tst accuracy: 0.8935
epoch: 12 softloss: 0.43807 trn accuracy: 0.894883 tst accuracy: 0.8948
epoch: 13 softloss: 0.427002 trn accuracy: 0.8968 tst accuracy: 0.8967
epoch: 14 softloss: 0.417156 trn accuracy: 0.898683 tst accuracy: 0.898
epoch: 15 softloss: 0.408329 trn accuracy: 0.9001 tst accuracy: 0.8983
epoch: 16 softloss: 0.40036 trn accuracy: 0.901367 tst accuracy: 0.8994
epoch: 17 softloss: 0.393121 trn accuracy: 0.902717 tst accuracy: 0.8998
epoch: 18 softloss: 0.386509 trn accuracy: 0.903983 tst accuracy: 0.9008
epoch: 19 softloss: 0.380439 trn accuracy: 0.90495 tst accuracy: 0.9019
epoch: 20 softloss: 0.374841 trn accuracy: 0.905683 tst accuracy: 0.903
epoch: 21 softloss: 0.369659 trn accuracy: 0.906917 tst accuracy: 0.9039
epoch: 22 softloss: 0.364844 trn accuracy: 0.90755 tst accuracy: 0.9041
epoch: 23 softloss: 0.360354 trn accuracy: 0.9084 tst accuracy: 0.9044
epoch: 24 softloss: 0.356156 trn accuracy: 0.909367 tst accuracy: 0.9054
epoch: 25 softloss: 0.352219 trn accuracy: 0.910083 tst accuracy: 0.9069
epoch: 26 softloss: 0.348518 trn accuracy: 0.9107 tst accuracy: 0.9072
epoch: 27 softloss: 0.345031 trn accuracy: 0.911233 tst accuracy: 0.9073
epoch: 28 softloss: 0.341739 trn accuracy: 0.911933 tst accuracy: 0.9077
epoch: 29 softloss: 0.338625 trn accuracy: 0.912567 tst accuracy: 0.9082
epoch: 30 softloss: 0.335674 trn accuracy: 0.91305 tst accuracy: 0.9092
epoch: 31 softloss: 0.332872 trn accuracy: 0.913467 tst accuracy: 0.9091
epoch: 32 softloss: 0.330209 trn accuracy: 0.9141 tst accuracy: 0.9093
epoch: 33 softloss: 0.327673 trn accuracy: 0.914433 tst accuracy: 0.9099
epoch: 34 softloss: 0.325256 trn accuracy: 0.914783 tst accuracy: 0.9103
epoch: 35 softloss: 0.322949 trn accuracy: 0.915167 tst accuracy: 0.9106
epoch: 36 softloss: 0.320744 trn accuracy: 0.915617 tst accuracy: 0.9108
epoch: 37 softloss: 0.318635 trn accuracy: 0.91585 tst accuracy: 0.9112
epoch: 38 softloss: 0.316615 trn accuracy: 0.91625 tst accuracy: 0.9113
epoch: 39 softloss: 0.314679 trn accuracy: 0.916683 tst accuracy: 0.9117
epoch: 40 softloss: 0.312821 trn accuracy: 0.917167 tst accuracy: 0.9122
epoch: 41 softloss: 0.311037 trn accuracy: 0.917383 tst accuracy: 0.9122
epoch: 42 softloss: 0.309322 trn accuracy: 0.9177 tst accuracy: 0.912
epoch: 43 softloss: 0.307672 trn accuracy: 0.918333 tst accuracy: 0.9126
epoch: 44 softloss: 0.306084 trn accuracy: 0.91875 tst accuracy: 0.9126
epoch: 45 softloss: 0.304554 trn accuracy: 0.919167 tst accuracy: 0.9127
epoch: 46 softloss: 0.30308 trn accuracy: 0.919483 tst accuracy: 0.9128
epoch: 47 softloss: 0.301657 trn accuracy: 0.9199 tst accuracy: 0.9128
epoch: 48 softloss: 0.300283 trn accuracy: 0.92005 tst accuracy: 0.9135
epoch: 49 softloss: 0.298956 trn accuracy: 0.920367 tst accuracy: 0.9141
epoch: 50 softloss: 0.297673 trn accuracy: 0.9206 tst accuracy: 0.9144
============================#
