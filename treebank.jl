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
        ("--lr"; arg_type=Float64; default=0.0001; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
		("--momentum"; arg_type=Float64; default=0.99; help="momentum")
		("--clip"; arg_type=Int; default=1; help="gradient clipping")
		("--unitnumber"; arg_type=Int; default=1000; help="number of units, sequence lenght")
		("--vocab"; arg_type=Bool; default=false; help="characters or words")
    end
    o = parse_args(s; as_symbols=true)
	o[:seed] = 123
    srand(o[:seed])
	
	train,valid,test= loaddata()
	
	if o[:vocab]	#words or characters as the vocabulary
		vocab, indices = wordVocabulary(train[1])
		trn = minibatch(vocab, split(train[1]), o[:batchsize])
		vld = minibatch(vocab, split(valid[1]), o[:batchsize])
		tst = minibatch(vocab, split(test[1]), o[:batchsize])
	else
		vocab, indices = charVocabulary(train[1])
		trn = minibatch(vocab, train[1], o[:batchsize])
		vld = minibatch(vocab, valid[1], o[:batchsize])
		tst = minibatch(vocab, test[1], o[:batchsize])
	end

	#initialize weights and states
	weights = initweights(o[:unitnumber], length(vocab), o[:lr])
	hidden_state = zeros(o[:unitnumber], o[:batchsize])
	cell_state = zeros(o[:unitnumber], o[:batchsize])
	
	#
	info("opts=",[(k,v) for (k,v) in o]...)
	info("vocabulary: ", length(vocab))
	#println(size(trn[1]))
	#report(epoch, input)=println((:epoch,epoch,:acc,accuracy(input, out)))
	#println((:loss, model(tst, state, out)))
	#
	
	#train
	for epoch = 1:o[:epochs]
		
	end
	generate(hidden_state, cell_state, weights, indices, n = 100)
end

function generate(hidden_state, cell_state, weights, indices, n = 100)
	input = zeros(1,length(vocab))
	
	for t 1:n
		out = predict(input, hidden_state, cell_state, weights)
		input = out
		print(reverseVocab(indices, out[j][i,:]))
	end
end

function train(inputs, hidden_state, cell_state, weights)
	for t = 1:lenght(inputs)
		loss_grad = lossgradient(inputs, hidden_state, cell_state, weights)
		
		for w in keys(model)
			update!(weights[k], loss_grad[k])
		end
	end
end

function predict(input, hidden_state, cell_state, weights)				#lstm architecture of one layer
	return lstm_cell(input, hidden_state, cell_state, weights)[3]
end

#bits-per-character error
#average_over_whole_data(-log_2(P(x_(t+1)|y_t))) 
#P -> softmax, select where x_t is 1
function bpc(input, ypred)
	p = sum(input .* softmax(ypred),2)
	return -sum(log2(p)) / size(ypred,1)
end

function model(inputs, hidden_state, cell_state, weights)
	sumloss = 0
	input = inputs[1]
	for t in length(inputs)
        ypred = predict(input, hidden_state, cell_state, weights)
        sumloss += bpc(ypred,inputs[t+1])	# error(Pr(x_t+1|y_t), x_t+1)
		input = inputs[t+1]
    end
    return sumloss
end

lossgradient = grad(model)

###									[[[1,0,0,0] [0,0,1,0] [1,0,0,0] [0,0,1,0]]]
###	minibatch(dict, abcdabcd, 4) -> [										  ]
###									[[[0,1,0,0] [0,0,0,1] [0,1,0,0] [0,0,0,1]]]
function minibatch(vocabulary, text, batchsize) ###for words split text
	vocab_lenght = length(vocabulary)
	batch_N = div(length(text),batchsize)
	data = [ falses(batchsize, vocab_lenght) for i=1:batch_N ]
	
	cidx = 0
    for c in text
		if isascii(c)
			idata = 1 + cidx % batch_N
			row = 1 + div(cidx, batch_N)
			row > batchsize && break
			col = vocabulary[c]
			data[idata][row,col] = 1
			cidx += 1
		end
    end
	#map(d->convert(KnetArray{Float32},d), data)
    return data
end

function initweights(hidden, vocab, winit)
    w = Dict()
    # your code starts here
	w[:xi] = winit*randn(hidden, vocab)
	w[:hi] = winit*randn(hidden, hidden)
	w[:ci] = winit*randn(hidden, hidden)
	w[:bi] = zeros(hidden)
	w[:xf] = winit*randn(hidden, vocab)
	w[:hf] = winit*randn(hidden, hidden)
	w[:cf] = winit*randn(hidden, hidden)
	w[:bf] = zeros(hidden)
	w[:xo] = winit*randn(vocab, vocab)
	w[:ho] = winit*randn(vocab, hidden)
	w[:co] = winit*randn(vocab, hidden)
	w[:bo] = zeros(vocab)
	w[:xc] = winit*randn(hidden, hidden)
	w[:hc] = winit*randn(hidden, hidden)
	w[:bc] = zeros(hidden)
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
			get!(vocab, word, length(vocab)+1)
			push!(indices, word)
		end
	end
	return vocab,indices
end

function charVocabulary(text)
	vocab = Dict{Char,Int}()
	indices = Vector{Char}()
	for c in text
		if isascii(c) 
			get!(vocab, c, length(vocab)+1) 
			push!(indices, c)
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