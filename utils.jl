
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
    it = sigm(weights[:xi]*input + weights[:hi]*hidden + weights[:ci] + weights[:bi])	#input gate
	ft = sigm(weights[:xf]*input + weights[:hf]*hidden + weights[:cf] + weights[:bf])	#forget gate
	ct = ft .* cell + it .* tanh(weights[:xc]*input + weights[:hc]*hidden + weights[:bc])	#cell state
	ot = sigm(weights[:xo]*input + weights[:ho]*hidden + weights[:co] + weights[:bo])	#output gate
    ht = ot .* tanh(ct)	#hidden state
    return (ht,ct, ot)
end
