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