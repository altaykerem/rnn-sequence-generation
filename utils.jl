Pkg.installed("Gadfly") == nothing && Pkg.add("Gadfly")
using Gadfly

function plotStrokes(strokes; title = "")
	println("Plotting... ",title)
	xses = cumsum(hcat(strokes...)[1,:])
	ys = cumsum(hcat(strokes...)[2,:])
	println(xses)
	println(ys)
	println(cumsum(hcat(strokes...)[3,:]))
	plot(x=xses, y=ys, Geom.point, Geom.line)
end

function softmax(ypred)
	ypred = ypred .- maximum(ypred, 1)
    prob = exp(ypred) ./ sum(exp(ypred), 1)
	return prob
end

function accuracy(ygold, yhat)
	correct = 0.0
	for j=1:size(ygold,1)
		for i=1:size(ygold[j], 2)
			correct += indmax(ygold[j][:,i]) == indmax(yhat[j][:, i]) ? 1.0 : 0.0
		end
	end
	return correct / (size(ygold, 1)*size(ygold[1],2))
end

function lstm_cell(input, hidden, cell, weights)
    it = sigm(weights[:xi]*input + weights[:hi]*hidden + weights[:ci]*cell .+ weights[:bi])	#input gate
	ft = sigm(weights[:xf]*input + weights[:hf]*hidden + weights[:cf]*cell .+ weights[:bf])	#forget gate
	ct = ft .* cell + it .* tanh(weights[:xc]*input + weights[:hc]*hidden .+ weights[:bc])	#cell state
	ot = sigm(weights[:xo]*input + weights[:ho]*hidden + weights[:co]*cell .+ weights[:bo])	#output gate
	ht = ot .* tanh(ct)	#hidden state
    return (ht,ct)
end

#initialize weights for lstm
function initlstmweights(hidden, nin, winit;atype=KnetArray{Float32})
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


