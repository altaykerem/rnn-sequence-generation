for p in ("Knet","ArgParse","LightXML")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet   
using ArgParse # To work with command line argumands
using LightXML

include("utils.jl")
#cd("Desktop\\Comp 441\\Sequence Generation project\\Sequence Project")
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
	lstmweights = [ initweights(o[:unitnumber], 3, o[:winit]; atype=o[:atype]) for i=1:o[:layersize] ]
	outputweights = initoutweights(o[:unitnumber], output_len, o[:winit]; atype=o[:atype])
	weights = (lstmweights, outputweights)
	hidden_state = [ convert(o[:atype], zeros(o[:unitnumber], o[:batchsize])) for i=1:o[:layersize] ]
	cell_state = [ convert(o[:atype], zeros(o[:unitnumber], o[:batchsize])) for i=1:o[:layersize] ]
	#params = initparams(weights;learn = o[:lr], clip = o[:clip], momentum = o[:momentum])
	
	println(size(trn[1]))
	println(trn[1])
	predict(trn[1], hidden_state, cell_state, weights, o[:layersize])
end

#get -log( P(x_(t+1)|y_t) ) 's
function lossfunc(M, input, output)
	sum = 0
	for i=1:M
		sum -= log(yt[:pi]*bivariate_prob(x1, x2, yt)) 	#add the bivariate probabilities
		bernoulli_eos = x3*yt[:eos] + (1-x3)*(1-yt[:eos]) #bernoulli end of sentence probabilities
		sum -= log(bernoulli_eos)
	end
	return sum
end

lossgradient = grad(lossfunc)

function bivariate_prob(x1, x2, yt)
	end_of_stroke = ypred[1]
	mv = reshape(ypred[2:end],6,M)
	pi, mu1, mu2, std1, std2, corr = (mv[1,:],mv[2,:],mv[3,:],mv[4,:],mv[5,:],mv[6,:])

	diff1 = x1-yt[:mu1]
	diff2 = x2-yt[:mu2]
	z = (diff1^2)/(yt[:std1]^2)+(diff2^2)/(yt[:std2]^2) - (2*yt[:corr]*diff1*diff2)/yt[:std1]*yt[:std2]
	k = 1 - yt[:corr]
	density = exp(-z/(2*k))
	density *= 1/(2*yt[:w]*yt[:std1]*yt[:std2]*sqrt(k))
	return density
end

# y is obtained by the network outputs(ypred)
function output_function(ypred, M)
	end_of_stroke = ypred[1]
	mv = reshape(ypred[2:end],6,M) #mixture vectors; {pi, mu(x1,x2), std(x1,x2), corr}M 
	w, mu1, mu2, std1, std2, corr = (mv[1,:],mv[2,:],mv[3,:],mv[4,:],mv[5,:],mv[6,:])
	
	#corresponding outputs of the predictions
	et = sigm(end_of_stroke)
	pi = softmax(w)
	std1 = exp(std1)
	std2 = exp(std2)
	corr = tanh(corr)
	
	return vcat(et, vec([pi mu1 mu2 std1 std2 corr]'))
end

#given the number of layers,n , returns network outputs
# ypred = [e, {pi, mu, std, corr}M] = b(y) + sum(W(hny) * h(nt))^N-n=1
function predict(input, hidden_state, cell_state, weights, n)
	hidden = zeros(size(hidden_state[1])) #hidden state passed between layers
	lstmw = first(weights)
	outw = weights[2]
	#iterate over the layers
	for i=1:n
		hidden,_ = lstm_cell(input, hidden_state[i].+ hidden, cell_state[i], lstmw[i])
	end
	return outw[:outw]*hidden .+ outw[:outb]
end

#initialize weights for lstm
function initweights(hidden, nin, winit;atype=KnetArray{Float32})
    w = Dict()
	#lstm weights
	w[:xi] = winit*randn(hidden, nin)		
	w[:hi] = winit*randn(hidden, hidden)	
	w[:ci] = winit*randn(hidden, hidden)	
	w[:bi] = zeros(hidden)
	w[:xf] = winit*randn(hidden, nin)		
	w[:hf] = winit*randn(hidden, hidden)	
	w[:cf] = winit*randn(hidden, hidden)	
	w[:bf] = zeros(hidden)
	w[:xo] = winit*randn(hidden, nin)		
	w[:ho] = winit*randn(hidden, hidden)	
	w[:co] = winit*randn(hidden, hidden)	
	w[:bo] = zeros(hidden)					
	w[:xc] = winit*randn(hidden, nin)		
	w[:hc] = winit*randn(hidden, hidden)	
	w[:bc] = zeros(hidden)					

	for k in keys(w)
		w[k] = convert(atype, w[k])
	end
    return w
end

#initialize weights for lstm
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

function minibatch(data, batchsize;atype=Array{Float32})
	batch_N = div(length(data),batchsize)
	sequence = [ zeros(3, batchsize) for i=1:batch_N ]

	idx = 0
    for p in data
		ibatch = 1 + idx % batch_N 
		ind = 1+div(idx, batch_N)
		ind > batchsize && break
		sequence[ibatch][:,ind] = p
		idx += 1
    end
    return map(d->convert(atype,d), sequence)
end

function loaddata(file; path="C:\\Users\\HP\\Desktop\\Comp 441\\Sequence Generation project\\$file")
	isfile(path)
	users = readdir(path)
	#lines = Any[]
	strokes = Any[]
	### get the input from file
	info("Reading data...")
	for form in users
		linedirs = readdir("$path\\$form")
		for linedir in linedirs
			line = readdir("$path\\$form\\$linedir")
			#push!(lines, line)
			for xml in line
				doc = parse_file("$path\\$form\\$linedir\\$xml")
				docroot = root(doc) # whiteboard session
				strokeset = find_element(docroot, "StrokeSet")
				for stroke in child_elements(strokeset)
					prevx, prevy = 0, 0
					for point in child_elements(stroke)
						x = parse(Int, attribute(XMLElement(point), "x"))
						y = parse(Int, attribute(XMLElement(point), "y"))
						push!(strokes, [x-prevx,y-prevy,0])		#keep x,y offsets from previous input
						prevx = x
						prevy = y
					end
					strokes[end][3] = 1							#1 indicating end of stroke
					free(stroke)
				end
			end
			break
		end
	end
	return strokes
end

main()