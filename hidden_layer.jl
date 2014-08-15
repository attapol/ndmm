# Hidden Layer implementation
# Feed-forward and update steps are inherited from the IntermediateLayer class
# we only have to implement a function that computes delta and the pretraining function

type HiddenLayer <: IntermediateLayer
    learning_rate::Float64
    lr_smoother::Float64
    num_units::Int64
    weights::Array
    sum_gradient_squared::Array
    activations#::Array{Float64,2}
    incoming_backprop_delta
    outgoing_backprop_delta::Array
    parents::Array{Layer}
    activation_fn
    activation_prime_fn
    add_bias
    clamped

    HiddenLayer(num_units::Int64, add_bias=true) = begin
        m = new(0.001, 0.00001, num_units, 
            Array[], Array[], [], [], Array[], Layer[], sigmoid, sigmoid_prime, add_bias, false)
        return m
    end
end

function initialize(this::HiddenLayer)
    initialize(this, true)
end

function compute_delta(this::HiddenLayer, target=None, target_weights=None)
    #println(this.activations)
    #println(this.activation_prime_fn(this.activations))
    #println(this.incoming_backprop_delta[:, 2:end])
    if this.add_bias
        return this.activation_prime_fn(this.activations) .* this.incoming_backprop_delta[:, 2:end]
    else
        return this.activation_prime_fn(this.activations) .* this.incoming_backprop_delta
    end
end

function get_activations(this::HiddenLayer)
    return this.activations
end

function pretrain(this::HiddenLayer, max_epochs)
    previous_error = Inf
    pos_hidden_states = None
    for epoch in 1:max_epochs
        # fix one of the input layers
        total_error = 0.0
        feed_forward(this)
        pos_hidden_activations = this.activations
        pos_hidden_probs = sigmoid(pos_hidden_activations)
        if pos_hidden_states == None
            pos_hidden_states = zeros(size(this.activations))
        end
        for i in 1:size(pos_hidden_probs, 1)
            for j in 1:size(pos_hidden_probs, 2)
                pos_hidden_states[i,j] = pos_hidden_probs[i,j] > rand(1)[1] ? 1 : 0
            end
        end
        #pos_hidden_states = map(x -> x > rand(1)[1]? 1: 0, pos_hidden_probs)
        
        num_cases = size(pos_hidden_activations, 1)
        for i in 1:length(this.parents)
            parent = this.parents[i]
            #positive phase
            if this.add_bias 
                pos_associations = [ones(1,num_cases)*pos_hidden_probs; 
                                get_activations(parent)'*pos_hidden_probs]
            else
                pos_associations = get_activations(parent)' * pos_hidden_probs
            end

            #negative phase
            neg_visible_activations = pos_hidden_states * this.weights[i]'
            neg_visible_probs = sigmoid(neg_visible_activations)
            if this.add_bias
                neg_visible_probs[:, 1] = 1 #fix the bias unit to one
            end
            neg_hidden_activations = neg_visible_probs * this.weights[i]
            neg_hidden_probs = sigmoid(neg_hidden_activations)
            neg_associations = neg_visible_probs' * neg_hidden_probs

            delta = (pos_associations - neg_associations) #/ this.num_cases
            this.sum_gradient_squared[i] += delta .^ 2
            adagrad_rate = this.learning_rate / 
                sqrt(this.lr_smoother + this.sum_gradient_squared[i])
            this.weights[i] += adagrad_rate .* delta
            if this.add_bias
                error = sum((get_activations(parent) - neg_visible_probs[:,2:end]) .^ 2)
            else
                error = sum((get_activations(parent) - neg_visible_probs) .^ 2)
            end
            total_error += error
        end
        #println ("Epoch $epoch : error is $total_error")
        if total_error > previous_error
            break
        else
            previous_error = total_error
        end
    end
    for i in 1:length(this.sum_gradient_squared)
        this.sum_gradient_squared[i][:,:] = 0.0
    end
    return previous_error
end

