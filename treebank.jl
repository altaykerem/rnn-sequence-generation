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
	data = loaddata("wsj")
	vocab = wordVocabulary(data)
	
	
	info("opts=",[(k,v) for (k,v) in o]...)
	#println(data)
	#println(length(split(data)))
	info("vocabulary: ", length(vocab))
	#for key in sort(collect(keys(vocab)))
    #    println("$key => $(vocab[key])")
    #end
	
	
end

function wordVocabulary(text)
	vocab = Dict{String, Int}()
	nextID = 0
	for word in split(text)
		nextID +=1
		isascii(word) && get!(vocab, word, nextID)
	end
	return vocab
end

function charVocabulary(text)
	vocab = Dict{Char,Int}()
	nextID = 0
	for c in text
		nextID +=1
		if isascii(c) get!(vocab, c, nextID) end
	end
    return vocab
end

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

main()