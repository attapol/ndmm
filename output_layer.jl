# Implementation of output layer and gated output Layer
# 
# GatedOutputLayer does not inherit from IntermediateLayer because
# it needs its own special feed_forward and update functions
type OutputLayer <: IntermediateLayer
    learning_rate::Float64
    lr_smoother::Float64
    num_units::Int64
    weights::Array
    sum_gradient_squared::Array
    activations#::Array{Float64,2}
    incoming_backprop_delta
    outgoing_backprop_delta::Array
    parents::Array{Layer}
    expert_probs
    gated::Bool
    activation_fn
    cost::Float64
    add_bias
    clamped

    OutputLayer(num_output_units) = begin
        return new(0.001, 0.00001, num_output_units, 
        Array[], Array[], [], [], Array[], Layer[], [], false, softmax, 0.0, true, false)
    end
end

function initialize(this::OutputLayer)
    initialize(this, false)
end

function compute_delta(this::OutputLayer, target, target_weights)
    if this.gated
        delta = target_weights .* (target - this.activations)
        for i in 1:size(delta, 1)
            delta[i,:] *= this.expert_probs[i]
        end
    else
        delta = target_weights .* (target - this.activations)
    end
    return delta
end

function compute_cost(this::Layer, target, target_weights)
    this.cost = sum(target_weights .* target .* -log(this.activations))
    return this.cost
end


#The implementation of Mixture of Experts model
# The layer also incorporate gating network, so forward and backward passes
# need to process the gating network as well

type GatedOutputLayer <: Layer
    learning_rate::Float64
    lr_smoother::Float64
    num_units::Int64
    activations#::Array{Float64,2}
    incoming_backprop_delta
    outgoing_backprop_delta::Array
    parents::Array{Layer} #input for determining gating probabilities
    experts::Array{Layer}
    gating_layer
    cost::Float64
    add_bias
    clamped

    GatedOutputLayer(num_output_units) = begin
        return new(0.001, 0.00001, num_output_units, 
            [], [], Array[], Layer[], Layer[], None, 0.0, true, false)
    end
end

function add_expert(this::GatedOutputLayer, expert_layer::OutputLayer)
    push!(this.experts, expert_layer)
    expert_layer.gated = true
end

function initialize(this::GatedOutputLayer)
    this.gating_layer = Net.OutputLayer(length(this.experts))
    this.gating_layer.parents = this.parents
    this.gating_layer.add_bias = this.add_bias
    this.gating_layer.clamped = this.clamped
    initialize(this.gating_layer, true)
end

function feed_forward(this::GatedOutputLayer)
    feed_forward(this.gating_layer)
    num_cases = size(this.gating_layer.activations, 1)
    this.activations = zeros(num_cases, this.num_units)

    for case_i in 1:size(this.activations, 1)
        for class_j in 1:size(this.activations, 2)
            for expert_k in 1:length(this.experts)
                this.activations[case_i, class_j] += 
                        this.gating_layer.activations[case_i, expert_k] * 
                        this.experts[expert_k].activations[case_i, class_j]
            end
        end
    end
end


function update(this::GatedOutputLayer, target, target_weights)
    # compute delta and propagate the probabilities to the experts
    # and update the weights of the gating network
    delta = zeros(size(this.gating_layer.activations))
    for expert in this.experts
        expert.expert_probs = zeros(size(delta, 1), 1)
    end
    for i in 1:size(delta, 1) #over cases
        target_ind = indmax(target[i, :])
        p_y_given_x = 0.0
        for j in 1:size(delta, 2) #over experts
            p_m_given_x = this.gating_layer.activations[i,j] 
            p_y_given_m_x = this.experts[j].activations[i, target_ind]
            p_y_given_x += p_m_given_x * p_y_given_m_x
        end

        for j in 1:size(delta, 2) #over experts
            p_m_given_x = this.gating_layer.activations[i,j] 
            p_y_given_m_x = this.experts[j].activations[i, target_ind]
            p_m_y_given_x = p_m_given_x * p_y_given_m_x
            #posterior over models
            p_m_given_y_x = p_m_y_given_x / p_y_given_x
            this.experts[j].expert_probs[i] = p_m_given_y_x
            delta[i,j] = target_weights[i] * (p_m_given_y_x - p_m_given_x)
            #delta[i,j] = target_weights[i] * (p_m_given_x - p_m_given_y_x)
        end
    end
    # update gating network
    update_weights_with_delta(this.gating_layer, delta)
end

