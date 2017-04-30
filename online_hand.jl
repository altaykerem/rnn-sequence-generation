for p in ("Knet","ArgParse","LightXML")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet   
using ArgParse # To work with command line argumands
using LightXML

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
		("--mixture";  arg_type=Int; default=20; help="number of mixture components")
		("--layersize";  arg_type=Int; default=3; help="number of layers")
		("--seqlength"; arg_type=Int; default=1; help="Number of steps to unroll the network for.")
		("--atype"; default=(gpu()>=0 ? KnetArray{Float32} : Array{Float32}); help="array type: Array for cpu, KnetArray for gpu")
    end

    o = parse_args(s; as_symbols=true)
	o[:seed] = 123
    srand(o[:seed])
    println("opts=",[(k,v) for (k,v) in o]...)
	
	###data
	data = loaddata("lineStrokes")[1:10]
	trn = minibatch(data, o[:batchsize])
	output_len = 1+(6*o[:mixture]) #eos + (20 weights, 40 means, 40 standard deviations and 20 correlations) were used in experiments
	
	#initialize weights and states
	layer1weights = initlstmweights(o[:unitnumber], 3, o[:winit]; atype=o[:atype])
	lstmweights = [ initlstmweights(o[:unitnumber], 3 + o[:unitnumber], o[:winit]; atype=o[:atype]) for i=2:o[:layersize] ]
	lstmweights = vcat(layer1weights, lstmweights)
	outputweights = initoutweights(o[:layersize]*o[:unitnumber], output_len, o[:winit]; atype=o[:atype])
	weights = (lstmweights, outputweights)
	hidden_state = [ convert(o[:atype], zeros(o[:unitnumber], o[:batchsize])) for i=1:o[:layersize] ]
	cell_state = [ convert(o[:atype], zeros(o[:unitnumber], o[:batchsize])) for i=1:o[:layersize] ]
	#params = initparams(weights;learn = o[:lr], clip = o[:clip], momentum = o[:momentum])

	ypred = predict(trn[1], hidden_state, cell_state, weights, o[:layersize])
	yout = output_function(ypred)
	lossfunc(trn[2], yout)
end

function train(inputs, hidden_state, cell_state, weights, prms; seq_length = 25)
	for t = 1:seq_length:length(inputs)-seq_length
		r = t:t+seq_length-1
		loss_grad = lossgradient(weights, inputs, hidden_state, cell_state; range = r)
		
		#update lstm weights
		for k in keys(weights[1])
			update!(weights[k], loss_grad[k], prms[k])
		end
		
		#update output weights
		for k in keys(weights[2])
			update!(weights[k], loss_grad[k], prms[k])
		end
	end
end

#sequence loss
function model(input, hidden_state, cell_state, weights; range = 1:length(inputs)-1)
	sumloss = 0
	input = inputs[first(range)]
	for t in range
        ypred = predict(input, hidden_state, cell_state, weights, length(hidden_state))
        sumloss += lossfunc(inputs[t+1], ypred)	# error(Pr(x_t+1|y_t), x_t+1)
		input = inputs[t+1]
    end
    return sumloss
end

lossgradient = grad(model)

#get -log( P(x_(t+1)|y_t) ) 's
function lossfunc(input, ypred)
	M = div((size(ypred,1)-1),6)
	eos = ypred[1,:]
	sum = 0
	for j=1:M
		pi_j = ypred[1+j,:]
		sum -= log(pi_j.*bivariate_prob(input, ypred, j)) 	#add the bivariate probabilities
	end
	bernoulli_eos = input[3,:].*eos + (1-input[3,:]).*(1-eos) #bernoulli end of sentence probabilities
	sum -= log(bernoulli_eos)
	return sum
end

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

#given the number of layers,n , returns network outputs
# ypred = [e, {pi, mu, std, corr}M] = b(y) + sum(W(hny) * h(nt))^N-n=1
function predict(input, hidden_state, cell_state, weights, n)
	lstmw = weights[1]
	outw = weights[2]
	#iterate over the layers
	hidden_state[1],cell_state[1] = lstm_cell(input, hidden_state[1], cell_state[1], lstmw[1])
	for i=2:n
		hidden_state[i],cell_state[i] = lstm_cell(vcat(input, hidden_state[i-1]), hidden_state[i], cell_state[i], lstmw[i])
	end
	return outw[:outw]*hcat(hidden_state) .+ outw[:outb]
end


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

function initparams(weights;learn = 0.1, clip=0, momentum=1.0)
    prms = Dict()
    for k in keys(weights)
		prms[k] = Momentum(;lr = learn, gclip = clip, gamma=0.9)
    end
    return prms
end

main()