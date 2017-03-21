for p in ("Knet","ArgParse","Compat")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet   
using ArgParse # To work with command line argumands
using Compat

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
	
	data = Any[]
	data,_,_= loaddata()
	vocab = charVocabulary(data[1])
	
	
	info("opts=",[(k,v) for (k,v) in o]...)
	info("vocabulary: ", length(vocab))
	
	minibatch_char(vocab, data[1], o[:batchsize])
	
end
#cd("Desktop\\Comp 441\\Sequence Generation project\\Sequence Project")

function minibatch_char(vocabulary, text, batchsize)
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

function minibatch_word(vocabulary, text, batchsize)
	vocab_lenght = length(vocabulary)
	batch_N = div(length(text),batchsize)
	data = [ falses(batchsize, vocab_lenght) for i=1:batch_N ]
	
	cidx = 0
    for word in split(text)
	if isascii(word)
        idata = 1 + cidx % batch_N
        row = 1 + div(cidx, batch_N)
        row > batchsize && break
        col = vocabulary[word]
        data[idata][row,col] = 1
        cidx += 1
	end
    end
	#map(d->convert(KnetArray{Float32},d), data)
    return data
end


function wordVocabulary(text)
	vocab = Dict{String, Int}()
	for word in split(text)
		isascii(word) && get!(vocab, word, length(vocab)+1)
	end
	return vocab
end

function charVocabulary(text)
	vocab = Dict{Char,Int}()
	for c in text
		if isascii(c) 
			get!(vocab, c, length(vocab)+1) 
		end
	end
    return vocab
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