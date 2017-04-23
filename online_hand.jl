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
        ("--epochs"; arg_type=Int; default=10; help="number of epoch ")
        ("--batchsize"; arg_type=Int; default=100; help="size of minibatches")
        ("--lr"; arg_type=Float64; default=0.0001; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
		("--momentum"; arg_type=Float64; default=0.99; help="momentum")
		("--clip"; arg_type=Int; default=1; help="gradient clipping")
		("--unitnumber"; arg_type=Int; default=400; help="number of units, sequence lenght")
		("--mixture";  arg_type=Int; default=20; help="number of mixture components")
		("--layersize";  arg_type=Int; default=3; help="number of layers")
    end

    o = parse_args(s; as_symbols=true)
	o[:seed] = 123
    srand(o[:seed])
    println("opts=",[(k,v) for (k,v) in o]...)
	
	data = loaddata("lineStrokes")
	
	
end

# y is obtained by the network outputs(ypred)
function getY(ypred, M)
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

# ypred = [e, {pi, mu, std, corr}M] = b(y) + sum(W(hny) * h(nt))^N-n=1
function predict(input, hidden_state, cell_state, weights, n)
	result = 0	#model output
	hidden = zeros(size(hidden_state[1])) #hidden state passed between layers
	#iterate over the layers
	for i=1:n
		hidden,_,out = lstm_cell(input, hidden_state[i].+ hidden, cell_state[i], weights[i])
		result += out
	end
	return result
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

function init_lstm_weights(hidden, nin, nout, winit)
    w = Dict()
    # your code starts here
	w[:xi] = winit*randn(hidden, nin)
	w[:hi] = winit*randn(hidden, hidden)
	w[:ci] = winit*randn(hidden, hidden)
	w[:bi] = zeros(hidden)
	w[:xf] = winit*randn(hidden, nin)
	w[:hf] = winit*randn(hidden, hidden)
	w[:cf] = winit*randn(hidden, hidden)
	w[:bf] = zeros(hidden)
	w[:xo] = winit*randn(nout, nin)
	w[:ho] = winit*randn(nout, hidden)
	w[:co] = winit*randn(nout, hidden)
	w[:bo] = zeros(nout)
	w[:xc] = winit*randn(hidden, nin)
	w[:hc] = winit*randn(hidden, hidden)
	w[:bc] = zeros(hidden)
    return w
end

function loaddata(file; path="C:\\Users\\HP\\Desktop\\Comp 441\\Sequence Generation project\\$file")
	isfile(path)
	users = readdir(path)
	#lines = Any[]
	strokes = Any[]
	
	### get the input from file
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
					prevx, prevy = 0
					s = Any[]
					for point in child_elements(stroke)
						x = attribute(XMLElement(point), "x")
						y = attribute(XMLElement(point), "y")
						push!(s, [x-prevx,y-prevy,0])		#keep x,y offsets from previous input
						prevx = x
						prevy = y
					end
					s[end][3] = 1							#1 indicating end of stroke
					push!(strokes, s)
					free(stroke)
				end
			end
		end
	end
	
	### 
	return strokes
end

main()