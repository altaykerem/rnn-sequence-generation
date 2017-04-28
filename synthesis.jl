for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet   
using ArgParse # To work with command line argumands

include("utils.jl")
include("online_hand.jl")
#cd("Desktop\\Comp 441\\Sequence Generation project\\Sequence Project")
function main(args="")
	s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=3; help="number of epoch ")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
		("--momentum"; arg_type=Float64; default=0.99; help="momentum")
		("--clip"; arg_type=Int; default=1; help="gradient clipping")
		("--unitnumber"; arg_type=Int; default=400; help="number of units, sequence lenght")
		("--mixture";  arg_type=Int; default=20; help="number of mixture components")
		("--seqlength"; arg_type=Int; default=1; help="Number of steps to unroll the network for.")
		("--atype"; default=(gpu()>=0 ? KnetArray{Float32} : Array{Float32}); help="array type: Array for cpu, KnetArray for gpu")
    end
	o = parse_args(s; as_symbols=true)
	
	#fixed variables
	alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,?!'"
	sentence = "Kerem Altay"
	batchsize = 1
	output_len = 1+(6*o[:mixture]) #eos + (20 weights, 40 means, 40 standard deviations and 20 correlations) were used in experiments
	vocab = charVocabulary(alphabet)
	
	println(onehot(vocab, 'c'))
	#initialize weights and states
	lstmweights = [ initweights(o[:unitnumber], 3, o[:winit]; atype=o[:atype]) for i=1:3 ]
	outputweights = initoutweights(o[:unitnumber], output_len, o[:winit]; atype=o[:atype])
	weights = (lstmweights, outputweights)
	hidden_state = [ convert(o[:atype], zeros(o[:unitnumber], batchsize)) for i=1:3 ]
	cell_state = [ convert(o[:atype], zeros(o[:unitnumber], batchsize)) for i=1:3 ]
	kappa = zeros(nmixtures)
	window = zeros(lenght(alphabet))
	#window weights
	wparam_size = 3*nmixtures
	window_w = winit*randn(wparam_size, hidden)
	window_b = zeros(wparam_size)
	
	c = characterSequence(vocab, sentence)
	#####predict
	#layer 1
	hidden,cell_state[1] = lstm_cell(input, hidden_state[1].+ hidden, cell_state[1], lstmw[1])
	#soft window, k-number of mixture elements = 3, t
	(alpha,beta, k) = windowParams(hidden, kappa)
	window = softWindow(alpha, beta, kappa, c)
	
	
end

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