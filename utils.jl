
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

b = zeros(10,5)
yh = Any[]
yg = Any[]
for i=1:10 b[i,mod(i,5)+1]=1 end
for i=1:41
	push!(yh, softmax(randn(10,5)))
	push!(yg, b)
end