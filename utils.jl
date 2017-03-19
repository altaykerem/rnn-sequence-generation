
function softmax(ypred)
	ypred = ypred .- maximum(ypred, 1)
    prob = exp(ypred) ./ sum(exp(ypred), 1)
	return prob
end

function accuracy(ygold, yhat)
	correct = 0.0
	for i=1:size(ygold, 2)
		correct += indmax(ygold[:,i]) == indmax(yhat[:, i]) ? 1.0 : 0.0
	end
	return correct / size(ygold, 2)
end