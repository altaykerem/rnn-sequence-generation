for p in ("Knet","ArgParse","Compat")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet   
using ArgParse # To work with command line argumands
using Compat

include("utils.jl")
#cd("Desktop\\Comp 441\\Sequence Generation project\\Sequence Project")
function main(args="")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=10; help="number of epoch ")
        ("--batchsize"; arg_type=Int; default=100; help="size of minibatches")
        ("--lr"; arg_type=Float64; default=0.3; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
		("--momentum"; arg_type=Float64; default=0.99; help="momentum")
		("--clip"; arg_type=Int; default=1; help="gradient clipping")
		("--unitnumber"; arg_type=Int; default=1000; help="number of units, sequence lenght")
		("--vocab"; arg_type=Bool; default=false; help="characters or words, false for character")
		("--seqlength"; arg_type=Int; default=25; help="Number of steps to unroll the network for.")
		("--atype"; default=(gpu()>=0 ? KnetArray{Float32} : Array{Float32}); help="array type: Array for cpu, KnetArray for gpu")
    end
    o = parse_args(s; as_symbols=true)
	o[:seed] = 123
    srand(o[:seed])
	
	#data related
	train_data,valid,test= loaddata()
	if o[:vocab]	#words or characters as the vocabulary
		vocab, indices = wordVocabulary(train_data[1])
		trn = minibatch(vocab, split(train_data[1]), o[:batchsize]; atype=o[:atype])
		vld = minibatch(vocab, split(valid[1]), o[:batchsize]; atype=o[:atype])
		tst = minibatch(vocab, split(test[1]), o[:batchsize]; atype=o[:atype])
	else
		vocab, indices = charVocabulary(train_data[1])
		trn = minibatch(vocab, train_data[1], o[:batchsize]; atype=o[:atype])[1:100]
		vld = minibatch(vocab, valid[1], o[:batchsize]; atype=o[:atype])[1:100]
		tst = minibatch(vocab, test[1], o[:batchsize]; atype=o[:atype])[1:100]
	end

	#initialize weights and states
	weights = initweights(o[:unitnumber], length(vocab), o[:winit]; atype=o[:atype])
	hidden_state = convert(o[:atype], zeros(o[:unitnumber], o[:batchsize]))
	cell_state = convert(o[:atype], zeros(o[:unitnumber], o[:batchsize]))
	params = initparams(weights;learn = o[:lr], clip = o[:clip], momentum = o[:momentum])
	
	#
	info("opts=",[(k,v) for (k,v) in o]...)
	info("vocabulary: ", length(vocab))
	loss(inputs) = model(weights, inputs, hidden_state, cell_state)
	acc(inputs) = accuracy(inputs[2:end] ,(map(b->predict(weights, b, hidden_state, cell_state), inputs[1:end-1])))
	report(epoch, inputs) = println((:epoch,epoch,:loss, loss(inputs), :acc, acc(inputs)))
	#

	
	#train
	info("Training... Size of traning: ", length(trn))
	validate_accuracy = 0
	report(0, trn)
	for epoch = 1:o[:epochs]
		#train
		train(trn, hidden_state, cell_state, weights, params;seq_length=o[:seqlength])
		report(epoch, trn)
			
		#validate
		newvalidate = acc(vld)
		if newvalidate<validate_accuracy
			println("starting to overfit")
			break
		end
		validate_accuracy = newvalidate
		println("val: ",validate_accuracy)
	end
	
	#test
	report("test", tst)
	#model output
	info("Model output")
	hidden_state = zeros(o[:unitnumber], 1)
	cell_state = zeros(o[:unitnumber], 1)
	generate(hidden_state, cell_state, weights, indices, 100)
end

function generate(hidden_state, cell_state, weights, indices, n = 100)
	input = zeros(length(indices),1)
	for t=1:n
		out = predict(weights, input, hidden_state, cell_state)
		input = out
		print(reverseVocab(indices, out))
	end
end

function train(inputs, hidden_state, cell_state, weights, prms; seq_length = 25)
	for t = 1:seq_length:length(inputs)-seq_length
		r = t:t+seq_length-1
		loss_grad = lossgradient(weights, inputs, hidden_state, cell_state; range = r)
		
		for k in keys(weights)
			update!(weights[k], loss_grad[k], prms[k])
		end
	end
end

function predict(weights, input, hidden_state, cell_state)				#lstm architecture of one layer
	ht = lstm_cell(input, hidden_state, cell_state, weights)[1]
	return output_layer(weights,ht)
end

function output_layer(weights, state)
	return weights[:outw]*state .+ weights[:outb]
end

#bits-per-character error
#average_over_whole_data(-log_2(P(x_(t+1)|y_t))) 
#P -> softmax, select where x_t is 1
function bpc(input, ypred)
	p = sum(input .* softmax(ypred),1)
	return -sum(log2(p)) / size(ypred,2)
end

#loss function
function model(weights, inputs, hidden_state, cell_state; range = 1:length(inputs)-1)
	sumloss = 0
	input = inputs[first(range)]
	for t in range
        ypred = predict(weights, input, hidden_state, cell_state)
        sumloss += bpc(inputs[t+1], ypred)	# error(Pr(x_t+1|y_t), x_t+1)
		input = inputs[t+1]
    end
    return sumloss
end

lossgradient = grad(model)

###									[[[1,0,0,0] [0,0,1,0] [1,0,0,0] [0,0,1,0]]]
###	minibatch(dict, abcdabcd, 4) -> [										  ]
###									[[[0,1,0,0] [0,0,0,1] [0,1,0,0] [0,0,0,1]]]
function minibatch(vocabulary, text, batchsize;atype=KnetArray{Float32})
	vocab_lenght = length(vocabulary)
	batch_N = div(length(text),batchsize)
	data = [ falses(vocab_lenght, batchsize) for i=1:batch_N ]
	
	cidx = 0
    for c in text
		if isascii(c)
			idata = 1 + cidx % batch_N
			col = 1 + div(cidx, batch_N)
			col > batchsize && break
			row = vocabulary[c]
			data[idata][row,col] = 1
			cidx += 1
		end
    end
    return map(d->convert(atype,d), data)
end

function initweights(hidden, vocab, winit;atype=KnetArray{Float32})
    w = Dict()
	#lstm weights
	w[:xi] = winit*randn(hidden, vocab)		
	w[:hi] = winit*randn(hidden, hidden)	
	w[:ci] = winit*randn(hidden, hidden)	
	w[:bi] = zeros(hidden)
	w[:xf] = winit*randn(hidden, vocab)		
	w[:hf] = winit*randn(hidden, hidden)	
	w[:cf] = winit*randn(hidden, hidden)	
	w[:bf] = zeros(hidden)
	w[:xo] = winit*randn(hidden, vocab)		
	w[:ho] = winit*randn(hidden, hidden)	
	w[:co] = winit*randn(hidden, hidden)	
	w[:bo] = zeros(hidden)					
	w[:xc] = winit*randn(hidden, vocab)		
	w[:hc] = winit*randn(hidden, hidden)	
	w[:bc] = zeros(hidden)					
	
	#output weights
	w[:outw] = winit*randn(vocab,hidden)
	w[:outb] = zeros(vocab)

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



function wordVocabulary(text)
	vocab = Dict{String, Int}()
	indices = Vector{String}()
	get!(vocab, " ", length(vocab)+1)
	push!(indices, " ")
	for word in split(text)
		if isascii(word) 
			if !haskey(vocab,word); push!(indices, word); end
			get!(vocab, word, length(vocab)+1)
		end
	end
	return vocab,indices
end

function charVocabulary(text)
	vocab = Dict{Char,Int}()
	indices = Vector{Char}()
	for c in text
		if isascii(c) 
			if !haskey(vocab,c); push!(indices, c); end
			get!(vocab, c, length(vocab)+1) 
		end
	end
    return vocab,indices
end

function reverseVocab(indices, vector)
	return indices[indmax(vector)]
end

function loaddata(path="C:\\Users\\HP\\Desktop\\Comp 441\\Sequence Generation project")
	train="";valid="";test = ""
	train = map((@compat readstring), Any["$path\\Mikolov_treebank_train.txt"])
	valid =  map((@compat readstring), Any["$path\\Mikolov_treebank_valid.txt"])
	test = map((@compat readstring), Any["$path\\Mikolov_treebank_test.txt"])
	
	return train,valid,test
end
#=== READS ACCORDING TO RAW TREEBANK FILE
function loaddata(file; path="C:\\Users\\HP\\Desktop\\Comp 441\\Sequence Generation project\\$file")
	text = "" 
	for c in readdir(path)
		for corpus in readdir("$path\\$c")
			stream = open("$path\\$c\\$corpus")
			text = text*readstring(stream)
			close(stream)
		end
	end
	return text
end
===#

main()