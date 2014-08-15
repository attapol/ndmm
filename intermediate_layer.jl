# Intermediate Layer implementation
# This is an abstract class 
#
# It has two 'subclasses' : HiddenLayer and OutputLayer
# This design is beneficial because the subclasses use the same implementation
# of feed_forward, update, and update_weights_with_delta 
# 
# OutputLayer, by definition, should not be an intermediate layer, but it can be
# if it's treated as a softmax layer or part of the mixture of expert architecture
# by connecting it to GatedOutputLayer
#
abstract Layer

abstract IntermediateLayer <: Layer

function initialize(this::Layer, randomize)
    if randomize
        initializer = randn
    else
        initializer = zeros
    end
        
    for parent in this.parents
        if this.add_bias
            weights = 0.1 * initializer(parent.num_units + 1, this.num_units)
        else
            weights = 0.1 * initializer(parent.num_units, this.num_units)
        end
        push!(this.weights, weights)
        push!(this.outgoing_backprop_delta, []) 
        push!(this.sum_gradient_squared, zeros(size(this.weights[end])))
    end
end

function feed_forward(this::IntermediateLayer)
    net_input = zeros(size(get_activations(this.parents[1]),1), this.num_units)
    for i in 1:length(this.parents)
        parent = this.parents[i]
        if this.add_bias
            for row in 1:size(net_input, 1)
                net_input[row,:] += this.weights[i][1,:]
            end
            net_input += get_activations(parent) * this.weights[i][2:end,:]
        else
            net_input += get_activations(parent) * this.weights[i]
        end
    end
    this.activations = this.activation_fn(net_input)
end

function update(this::IntermediateLayer, target=None, target_weights=None)
    #this function completes more than one thing
    # 1) compute delta
    # 2) compute gradient, adagrad update rate, and update the weights
    # 3) compute the error to backpropagate to the parent layers
    delta = compute_delta(this, target, target_weights)
    update_weights_with_delta(this, delta)
end

function update_weights_with_delta(this::Layer, delta)
    for i in 1:length(this.parents)
        parent = this.parents[i]
        if this.add_bias
            gradient = [ones(1,size(delta,1)) * delta; get_activations(parent)' * delta]
        else
            gradient = get_activations(parent)' * delta
        end
        gradient /= size(delta, 1)
        this.sum_gradient_squared[i] += gradient .^ 2
        adagrad_rate = this.learning_rate / sqrt(this.lr_smoother + this.sum_gradient_squared[i])
        if typeof(this.parents[i]) != Net.InputLayer
            this.outgoing_backprop_delta[i] = delta * this.weights[i]' 
            parent.incoming_backprop_delta = this.outgoing_backprop_delta[i]
        end
        if !this.clamped 
            this.weights[i] += adagrad_rate .* gradient
        end
    end
end
