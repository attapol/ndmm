# Multi-layer neural net
# 
# This implementation allows any arbitrary connection between layers. 
# but it only allows one (error-initiated) output layer for now
#

type MultiNet
    layers
    add_bias
    input_layer_indices::Array{Int64,1}
    output_layer_index::Int64
    connections::Dict{Int64, Array{Int64,1}}
    reversed_connections::Dict{Int64, Array{Int64,1}}
    ordering::Array{Int64, 1}
    io

    MultiNet() = new(Layer[], true, Int64[], 0, 
        Dict{Int64,Array{Int64,1}}(), Dict{Int64, Array{Int64,1}}(), Int64[], STDOUT)
end

function add_input_layer(this::MultiNet, layer)
    layer_index = add_layer(this, layer)
    push!(this.input_layer_indices, layer_index)
    return layer_index
end

function add_output_layer(this::MultiNet, layer)
    layer_index = add_layer(this, layer)
    this.output_layer_index = layer_index
    return layer_index
end

function add_layer(this::MultiNet, layer)
   push!(this.layers, layer)
   return length(this.layers)
end

function connect(this::MultiNet, from_index, to_index)
    # initialize the weights in the destination layer
    # instead of copying the whole thing
    # 
    # this function should be called in the proper order 
    # if a -> b -> c 
    # then connect(a,b) and connect(b,c)
    # you cannot do connect(b,c) and then connect(a,b)
    # the initialization will get messed up because we don't do any memory allocation 
    # for the source layer
    #
    if !haskey(this.reversed_connections, to_index)
        this.reversed_connections[to_index] = []
    end
    push!(this.reversed_connections[to_index], from_index)

    if !haskey(this.connections, from_index)
        this.connections[from_index] = []
    end
    push!(this.connections[from_index], to_index)
    push!(this.layers[to_index].parents, this.layers[from_index])
end

function initialize(this::MultiNet)
    #Call this function after the connections have been set up
    i = 0
    for layer in this.layers
        initialize(layer)
        i +=1
    end
end

function compute_ordering(this::MultiNet)
    this.ordering = [i for i in this.input_layer_indices]
    current_index = 1
    while length(this.ordering) != length(this.layers)
        is_ready = !haskey(this.reversed_connections, current_index) ||
            issubset(this.reversed_connections[current_index], this.ordering) 
        if is_ready & !(current_index in this.ordering)
           push!(this.ordering, current_index)
        end
        current_index = (current_index % length(this.layers)) +1
    end
    @assert this.ordering[end]==this.output_layer_index "The ordering is $ordering. The output layer index is $(this.output_layer_index)"
    return this.ordering
end

function feed_input_layer(this::MultiNet, data)
    # Feed data into input layers before running feed forward and backpropagation
    num_input_data = length(data)
    num_input_layer = length(this.input_layer_indices)
    @assert num_input_data==num_input_layer "There are $num_input_layer input layers. Got $num_input_data input data matrices"
    for i in 1:num_input_data
        input_layer = this.layers[this.input_layer_indices[i]]
        input_layer.activations = data[i]
        input_layer.data_range = 1:size(data[i], 1)
    end
end

function feed_forward(this::MultiNet, data_range)
    #this version assumes that the data has already been set
    for layer_index in this.input_layer_indices
        this.layers[layer_index].data_range = data_range
    end
    for layer_index in this.ordering
        feed_forward(this.layers[layer_index])
    end
end


function forward_backprop_update(this::MultiNet, data_range, target, target_weights,
        learning_rate, lr_smoother)
    feed_forward(this, data_range)
    ordering = reverse(this.ordering)
    @assert ordering[1]==this.output_layer_index "The first layer to backprop must be output layer"
    for i in ordering
        this.layers[i].learning_rate = learning_rate
        this.layers[i].lr_smoother = lr_smoother
        update(this.layers[i], target[data_range, :], target_weights[data_range, :])
        if i == this.output_layer_index
            compute_cost(this.layers[i], target[data_range, :], target_weights[data_range, :])
        end
    end
    return this.layers[this.output_layer_index].cost
end

function train(this::MultiNet, data, target, dev_data, dev_target, test_data, test_target, 
        learning_rate::Float64, lr_smoother::Float64, max_epochs::Int64=20, batch_size::Int64=0)
    target_weights = ones(size(target, 1), 1)
    return train(this, data, target, target_weights, dev_data, dev_target, test_data, test_target, 
        learning_rate, lr_smoother, max_epochs, batch_size)
end

function train(this::MultiNet, data, target, target_weights, dev_data, dev_target,
    test_data, test_target,
        learning_rate::Float64, lr_smoother::Float64, max_epochs::Int64=20, batch_size::Int64=0)
    compute_ordering(this)
    losses = zeros(max_epochs)
    dev_confusion_matrices = Array[]
    test_confusion_matrices = Array[]
    num_cases = size(data[1], 1)
    for epoch in 1:max_epochs
        tic()
        feed_input_layer(this, data)
        if batch_size == 0
            cost = forward_backprop_update(this, 1:num_cases, target, target_weights,
                learning_rate, lr_smoother)
        else
            num_batches = num_cases / batch_size
            cost = 0.0
            subiteration_time = 0.0
            for batch_index in 0:num_batches-1
                tic()
                start_index = batch_index * batch_size + 1
                end_index = min((batch_index + 1) * batch_size, num_cases)
                cost += forward_backprop_update(this, start_index:end_index, target, target_weights, learning_rate, lr_smoother)
                subiteration_time += toq()
                if batch_index % 100 == 0
                    println(this.io, "Subiteration $epoch.$batch_index Time $subiteration_time")
                    flush(this.io)
                    subiteration_time = 0
                end
            end
        end
        iteration_time = toq()
        dev_mean_f1, dev_accuracy, dev_confusion_matrix = evaluate(this, dev_data, dev_target, true)
        test_mean_f1, test_accuracy, test_confusion_matrix = evaluate(this, test_data, test_target, false)
        push!(dev_confusion_matrices, dev_confusion_matrix)
        push!(test_confusion_matrices, test_confusion_matrix)
        println (this.io, "Iteration $epoch : loss $cost, F1 $dev_mean_f1, Accuracy $dev_accuracy, Time $iteration_time")
        flush(this.io)
        losses[epoch] = cost
    end
    #draw(D3("plot.js", 6inch, 6inch),plot(x=1:max_epochs, y=losses))
    return losses, dev_confusion_matrices, test_confusion_matrices
end

function pretrain(this::MultiNet, num_iterations::Int64)
    compute_ordering(this)
    for i in this.ordering
        if typeof(this.layers[i]) == Net.HiddenLayer
            print (this.io, "Pretraining layer $i ")
            flush(this.io)
            cost = pretrain(this.layers[i], num_iterations)
            println (this.io, ": final cost is $cost")
            flush(this.io)
        end
    end
end

function evaluate(this::MultiNet, test_data, test_target, verbose=false)
    if test_data == None || test_target == None
        return  NaN, NaN, None
    end
    num_cases = size(test_target, 1)
    feed_input_layer(this, test_data)
    feed_forward(this, 1:num_cases)
    output_layer = this.layers[this.output_layer_index]
    activations = output_layer.activations[1:num_cases,:]
    num_output_units = size(activations, 2)
    confusion_matrix = zeros(num_output_units, num_output_units)
    for i in 1:num_cases
        prediction = indmax(activations[i,:])
        answer = indmax(test_target[i, :])
        confusion_matrix[answer, prediction] += 1.0
    end
    println(this.io)
    precisions = [confusion_matrix[i,i] / sum(confusion_matrix[:,i]) for i in 1:num_output_units]'
    recalls = [confusion_matrix[i,i] / sum(confusion_matrix[i,:]) for i in 1:num_output_units]'
    f1s = 2 * (precisions .* recalls) ./ (precisions + recalls) 
    if verbose
        print(this.io, "Precision $precisions")
        print(this.io, "Recall $recalls")
        print(this.io, "F1 $f1s")
    end
    return (mean(f1s), trace(confusion_matrix) / sum(confusion_matrix), confusion_matrix)
end

