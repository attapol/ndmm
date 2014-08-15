# All activation functions and their derivatives
function sigmoid(net_input)
    return 1 / (1 + exp(-net_input))
end

function sigmoid_prime(activations)
    return activations .* (1 - activations)
end

function relu(net_input)
    # rectified linear unit 
    activations = zeros(size(net_input))
    for i in 1:size(net_input, 1)
        for j in 1:size(net_input, 2)
            activations[i, j] = max(net_input[i,j], 0)
        end
    end
    return activations
end 

function relu_prime(activations)
    derivatives = zeros(size(activations))
    for i in 1:size(activations, 1)
        for j in 1:size(activations, 2)
            derivatives[i,j] = activations[i,j] > 0 ? 1 : 0
        end
    end
    return derivatives
end

function softmax(x)
    activations = exp(x)
    for i in 1:size(activations, 1)
        activations[i,:] = activations[i,:] ./ sum(activations[i,:])
    end
    return activations
end

