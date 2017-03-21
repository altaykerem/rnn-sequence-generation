for p in ("Knet","ArgParse","Compat")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet   
using ArgParse # To work with command line argumands
using Compat

include("utils.jl")

function main(args="")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=10; help="number of epoch ")
        ("--batchsize"; arg_type=Int; default=100; help="size of minibatches")
        ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
    end

    o = parse_args(s; as_symbols=true)
	o[:seed] = 123
    srand(o[:seed])
	
	data,valid,test= loaddata()
	vocab, indices = wordVocabulary(data[1])
	trn = minibatch(vocab, split(data[1]), o[:batchsize])
	vld = minibatch(vocab, split(valid[1]), o[:batchsize])
	tst = minibatch(vocab, split(test[1]), o[:batchsize])
	out = Any[]
	
	info("opts=",[(k,v) for (k,v) in o]...)
	info("vocabulary: ", length(vocab))
	report(epoch, input)=println((:epoch,epoch,:acc,accuracy(input, out)))
	
	
	state = tst[1]
	
	println((:loss, model(tst, state, out)))
	report(0,tst)
	for j=1:size(out,1)
		for i=1:size(out[1],1)
			print(reverseVocab(indices, out[j][i,:])) 
		end
		println()
	end
end
#cd("Desktop\\Comp 441\\Sequence Generation project\\Sequence Project")

###creates a random output independant from input
function random_cell(input,state;lr=0.1)
	return softmax(lr*randn(size(input)))
end

#bits-per-character error
#average_over_whole_data(-log_2(P(x_(t+1)|y_t))) 
#P -> softmax, select where x_t is 1
function bpc(input, ypred)
	p = sum(input .* softmax(ypred),2)
	return -sum(log2(p)) / size(ypred,1)
end

function model(inputs, state, out)
	sumloss = 0
	for t in 1:length(inputs)
        output = random_cell(inputs[t],state)
        sumloss += bpc(output,inputs[t])
		push!(out, output)
    end
    return sumloss
end

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