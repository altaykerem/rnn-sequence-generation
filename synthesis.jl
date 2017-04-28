for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet   
using ArgParse # To work with command line argumands

include("utils.jl")
include("online_data_loader.jl")

function main(args="")
	s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=3; help="number of epoch ")
		("--batchsize"; arg_type=Int; default=5; help="size of minibatches")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
		("--momentum"; arg_type=Float64; default=0.99; help="momentum")
		("--clip"; arg_type=Int; default=1; help="gradient clipping")
		("--unitnumber"; arg_type=Int; default=400; help="number of units, sequence lenght")
		("--mixture";  arg_type=Int; default=20; help="number of mixture components at output layer")
		("--kmixture";  arg_type=Int; default=10; help="number of mixture components in window parameters")
		("--seqlength"; arg_type=Int; default=20; help="Number of steps to unroll the network for.")
		("--atype"; default=(gpu()>=0 ? KnetArray{Float32} : Array{Float32}); help="array type: Array for cpu, KnetArray for gpu")
    end
	o = parse_args(s; as_symbols=true)
	
	#fixed variables
	alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,?!'"
	sentence = "Kerem Altay" #sequence to synthesize
	output_len = 1+(6*o[:mixture]) #eos + (20 weights, 40 means, 40 standard deviations and 20 correlations) were used in experiments
	window_size = lenght(alphabet)
	vocab = charVocabulary(alphabet)
	hsize = o[:unitnumber]
	c = characterSequence(vocab, sentence)
	
	###initialize weights
		#lstm1 (x, prev, prevWindow) -> h
	lstm_layer1_weights = initlstmweights(hsize, 3 + window_size, o[:winit]; atype=o[:atype])
	windowweights = initwindowweights(o[:unitnumber], 3*o[:kmixture]; atype=o[:atype])
		#lstmN (x, prevhN, hN-1, window) -> h
	lstm_layer2_weights = initlstmweights(hsize, 3 + hsize + window_size, o[:winit]; atype=o[:atype])
	lstm_layer3_weights = initlstmweights(hsize, 3 + hsize + window_size, o[:winit]; atype=o[:atype])
	outputweights = initoutweights(hsize, output_len, o[:winit]; atype=o[:atype])
	weights = (lstm_layer1_weights, windowweights, lstm_layer2_weights,lstm_layer3_weights, outputweights)
	
	###initialize states
	hidden_state = [ convert(o[:atype], zeros(o[:unitnumber], batchsize)) for i=1:3 ]
	cell_state = [ convert(o[:atype], zeros(o[:unitnumber], batchsize)) for i=1:3 ]
	kappa = zeros(nmixtures)
	window = zeros(window_size)


	#####predict
	#layer 1
	hidden,cell_state[1] = lstm_cell(input, hidden_state[1].+ hidden, cell_state[1], lstmw[1])
	#soft window, k-number of mixture elements = 3, t
	(alpha,beta, k) = windowParams(hidden, kappa)
	window = softWindow(alpha, beta, kappa, c)
	
	
end

###Soft window
#Mixture of K gaussian functions is used with 3 parameters alpha beta and kappa
function softWindow(alpha, beta, kappa, c)
	w = zeros(length(c))
	for u=1:length(c)
		w += window_weight_func(u,alpha,beta, k)*c[u]
	end
	return w
end

function windowParams(input, prevk)
	nmixtures = size(input,1)/3
	alphahat, betahat, khat = (input[1:nmixtures,:],input[1+nmixtures:2nmixtures,:],input[1+2nmixtures:3nmixtures,:])
	alpha = exp(alphahat)
	beta = exp(betahat)
	k = prevk + exp(khat)
	return alpha,beta, k
end

function window_weight_func(u,alpha,beta, k)
	return sum(alpha .* exp(-beta .* (k .- u).^2),1)
end
###End soft window

###Output mixture density model
#Given one of the mixtures,j range(1,M), returns the bivariate probability density function
function bivariate_prob(input, ypred, j)
	M = div((size(ypred,1)-1),6)
	mu1, mu2, std1, std2, corr = (ypred[M+1+j,:],ypred[2M+1+j,:],ypred[3M+1+j,:],ypred[4M+1+j,:],ypred[5M+1+j,:])

	diff1 = input[1,:]-mu1
	diff2 = input[2,:]-mu2
	z = (diff1.^2)./(std1.^2)+(diff2.^2)./(std2.^2) - (2*corr.*diff1.*diff2)./(std1.*std2)
	k = 1 - corr.^2
	density = exp(-z./(2*k))
	density = density./(2*pi.*std1.*std2.*sqrt(k))
	return density
end

# y is obtained by the network outputs(ypred)
function output_function(ypred)
	out = zeros(size(ypred))
	M = div((size(ypred,1)-1),6)
	#mixture vectors; eos{pi, mu(x1,x2), std(x1,x2), corr}M 
	end_of_stroke = ypred[1,:]
	w, mu1, mu2, std1, std2, corr = (ypred[2:M+1,:],ypred[2+M:2M+1,:],ypred[2M+2:3M+1,:],ypred[3M+2:4M+1,:],ypred[4M+2:5M+1,:],ypred[5M+2:6M+1,:])
	
	#corresponding outputs of the predictions
	out[1,:] = sigm(end_of_stroke)
	out[2:M+1,:] = softmax(w)				#pi
	out[3M+2:4M+1,:] = exp(std1)
	out[4M+2:5M+1,:] = exp(std2)
	out[5M+2:6M+1,:] = tanh(corr)
	
	return out
end
###End mixture density model

#initialize weights for output
function initoutweights(hidden, nout, winit;atype=KnetArray{Float32})
    w = Dict()				
	
	#output weights
	w[:outw] = winit*randn(nout,hidden)
	w[:outb] = zeros(nout)

	for k in keys(w)
		w[k] = convert(atype, w[k])
	end
    return w
end
#initialize weights for window
function initwindowweights(hidden, wparam_size, winit;atype=KnetArray{Float32})
    w = Dict()				
	
	#output weights
	w[:window_w] = winit*randn(wparam_size, hidden)
	w[:window_b] = zeros(wparam_size)

	for k in keys(w)
		w[k] = convert(atype, w[k])
	end
    return w
end

function characterSequence(vocab, sequence)
	seq = Any[]
	for c in sequence
		push!(seq, onehot(vocab, c))
	end
	return seq
end

function onehot(vocabulary, char;atype=Array{Float32})
	vocab_lenght = length(vocabulary)
	vec = falses(vocab_lenght)
	vec[vocabulary[char]] = 1
    return convert(atype,vec)
end

function charVocabulary(text)
	vocab = Dict{Char,Int}()
	for c in text
		get!(vocab, c, length(vocab)+1) 
	end
    return vocab
end

main()